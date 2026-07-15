import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from eval_envs.model.base import BaseModel
from eval_envs.model.nn_modules import ConditionalUnet1D, get_image_encoder, SpatialSoftmax


class DiffusionPolicyUnet(BaseModel):
    def __init__(
            self,
            shape_meta,
            obs_encoder,
            noise_scheduler,
            obs_horizon,
            action_exec_horizon,
            action_pred_horizon,
            image_aug,
            down_dims,
            kernel_size,
            n_groups,
            diffusion_step_embed_dim,
            weight_ratio=10,
            warmup_steps=20000,
            include_text: bool = False,
            text_embed_dim: int = 512,     # CLIP
            text_proj_dim: int = 64,       # target projection dim
            text_mlp_hidden: int = 64,    # small hidden size
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_dim = shape_meta.action.shape[0]
        self.state_dim = shape_meta.state.shape[0]
        # self.obs_feature_dim = obs_encoder.output_dim + \
        #     self.state_dim  # just concate them, 64 + 2 = 66
        self.include_text = include_text
        self.text_embed_dim = text_embed_dim
        self.text_proj_dim = text_proj_dim
        C = shape_meta.image.shape[0]   # 6 in your config
        K = C // 3                      # 2 cameras
        self.obs_feature_dim = (obs_encoder.output_dim * K) + self.state_dim
        if self.include_text:
            # tiny projector: 512 -> 16 (or 24)
            self.text_proj = nn.Sequential(
                nn.LayerNorm(self.text_embed_dim),
                nn.Linear(self.text_embed_dim, text_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(text_mlp_hidden, self.text_proj_dim),
            )
            self.obs_feature_dim += self.text_proj_dim
        self.obs_horizon = obs_horizon
        self.action_exec_horizon = action_exec_horizon
        self.action_pred_horizon = action_pred_horizon
        self.global_cond_dim = self.obs_horizon * self.obs_feature_dim
        
        self.noise_scheduler = noise_scheduler
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.down_dims = down_dims
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.weight_ratio = weight_ratio
        self.warmup_steps = warmup_steps
        self.counter = 0

        self.policy = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler.num_train_timesteps,
            beta_start=noise_scheduler.beta_start,
            beta_end=noise_scheduler.beta_end,
            beta_schedule=noise_scheduler.beta_schedule,
            variance_type=noise_scheduler.variance_type,
            clip_sample=noise_scheduler.clip_sample,
            prediction_type=noise_scheduler.prediction_type,
        )
        self.num_inference_steps = noise_scheduler.num_inference_steps

        self.image_encoder = get_image_encoder(obs_encoder.name)
        H, W = shape_meta.image.shape[1:]
        crop_ratio = image_aug.crop_ratio
        H_crop, W_crop = int(H * crop_ratio), int(W * crop_ratio)
        dummy_input = torch.zeros(size=(1, 3, H_crop, W_crop))
        with torch.inference_mode():
            dummy_feature_map = self.image_encoder(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])

        self.image_pool = nn.Sequential(
            # nn.Linear(512, config.model.obs_encoder.output_dim),
            SpatialSoftmax(input_shape=feature_map_shape,
                           num_kp=obs_encoder.num_kp),
            nn.Flatten(start_dim=1),
            nn.ReLU(),  # dim = 64
        )
        self.img_tf_train = v2.Compose([
            v2.RandomCrop(size=(H_crop, W_crop)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.img_tf_val = v2.Compose([
            v2.CenterCrop(size=(H_crop, W_crop)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        text_emb: torch.Tensor | None = None,
    ):
        """
        image:
        - (B, T, C, H, W) where C=3 (single cam) or C=3*K (K cams concatenated along channel)
            e.g. two cams -> C=6
        state: (B, T, state_dim)
        text_emb (optional): (B, T, 512)
        """
        B, T, C, H, W = image.shape
        assert state.shape == (B, T, self.state_dim), (
            f"state shape {state.shape} != {(B, T, self.state_dim)}"
        )

        # number of cameras packed in channel dimension
        if C % 3 != 0:
            raise ValueError(f"Expected image channels divisible by 3, got C={C}")
        K = C // 3  # cameras

        # (B, T, C, H, W) -> (B, T, K, 3, H, W)
        image = image.view(B, T, K, 3, H, W)

        # flatten cameras into batch for encoding: (B*T*K, 3, H, W)
        image = image.reshape(B * T * K, 3, H, W)

        # flatten state as (B*T, Ds)
        state_flat = state.reshape(B * T, self.state_dim)

        # augment/normalize images
        if self.training:
            image = self.img_tf_train(image)
        else:
            image = self.img_tf_val(image)

        # encode images: (B*T*K, F)
        img_feat = self.image_pool(self.image_encoder(image))
        F = img_feat.shape[-1]

        # unflatten camera dimension and concatenate camera features:
        # (B*T*K, F) -> (B*T, K, F) -> (B*T, K*F)
        img_feat = img_feat.view(B * T, K, F).reshape(B * T, K * F)

        feats = [img_feat, state_flat]

        if self.include_text:
            if text_emb is None:
                raise ValueError("text_emb must be provided when include_text is True")
            # ensure text_emb is (B,T,512)
            if text_emb.shape[:2] != (B, T):
                raise ValueError(f"text_emb leading dims {text_emb.shape[:2]} != {(B,T)}")
            text_flat = text_emb.reshape(B * T, self.text_embed_dim)  # (B*T, 512)
            text_small = self.text_proj(text_flat)                    # (B*T, text_proj_dim)
            feats.append(text_small)

        # final obs feature per timestep: (B*T, K*F + Ds [+ text])
        obs_feature = torch.cat(feats, dim=-1)

        # global conditioning: (B, T*(...))
        global_cond = obs_feature.view(B, -1)
        return global_cond

    def compute_loss(self,
                     batch_dict: dict[str, torch.Tensor]
                     ) -> tuple[torch.Tensor, dict]:
        image = batch_dict['image']
        state = batch_dict['state']
        action = batch_dict['action']
        text_emb = batch_dict.get('text_emb', None)
        global_cond = self.forward(image, state,text_emb=text_emb)
        trajectory = action
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            # (B, action_pred_horizon, action_dim)
            trajectory, noise, timesteps)

        # Predict the noise residual
        pred = self.policy(noisy_trajectory, timesteps,
                           global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        
        loss = F.mse_loss(pred, target)
        return loss, {}

    # ========= inference  ============

    def conditional_sample(
        self,
        condition_data, condition_mask,
        global_cond=None,
    ):
        model = self.policy
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory, t, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self,
        obs_dict, policy_info
    ):

        image = obs_dict['image']
        state = obs_dict['state']
        text_emb = obs_dict.get('text_emb', None)
        global_cond = self.forward(image, state, text_emb=text_emb)
        B = global_cond.shape[0]
        T = self.action_pred_horizon
        Da = self.action_dim
        device = global_cond.device
        dtype = global_cond.dtype

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        action_pred = self.conditional_sample(
            cond_data, cond_mask,
            global_cond=global_cond)

        start = self.obs_horizon - 1
        end = start + self.action_exec_horizon
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'state': state,
        }
        return result, policy_info