import copy
import os
import json
import random
import shutil
import hydra
import omegaconf
import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.optim import AdamW
from eval_envs.utils.pytorch_util import dict_apply
from diffusers.training_utils import EMAModel
from eval_envs.utils.dataloader_util import InfiniteRandomSampler
from eval_envs.utils.distributed_util import remove_prefix

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def make_env_runner_from_cfg(cfg):
    return hydra.utils.instantiate(cfg)

def get_exp_id(config):
    if config.exp_id is None:
        s = f"bs{config.dataloader.batch_size}_obs{config.obs_horizon}_exe{config.action_exec_horizon}_pre{config.action_pred_horizon}"

        if config.norm_type in ["minmax", "mm"]:
            s += "_mmnorm"
        elif config.norm_type in ["standard", "std"]:
            s += "_stdnorm"
        elif config.norm_type in ["quantile", "q"]:
            s += "_qnorm"
        else:
            raise ValueError(f"Invalid norm type: {config.norm_type}")

        if config.use_delta_action:
            s += "_delta"

        if not config.training.use_ema:
            s += "_noema"
        
    
        if  config.model.get("use_chunk_norm_weighted_loss", False):
            s += f"_lcn-r{config.model.weight_ratio}"
            if config.model.get("warmup_steps", None) is not None:
                s += f"-w{config.model.warmup_steps}"
                
        if "dp" in config.name:
            s = "dp_" + s
        elif "fm" in config.name:
            s = "fm_" + s
        
        
        if hasattr(config, "noise_generator"):
            s += "_ns-" + config.noise_generator.name
            if config.model.get("use_hybrid_loss", False):
                s += "-hyb"
            if not config.model.get("share_obs_encoder", True):
                s += "-dua"
            if config.model.get("stop_obs_encoder_grad", False):
                s += "-sgd"
            
            if config.model.get("kl_weight_base", 1e-3) < 1e-8:
                s += "-nkl"
            else:
                s += f"-klw{config.model.kl_weight_base:.0e}-r{config.model.kl_weight_horizon_ratio}-s{config.model.kl_annealing_steps:.0e}"
            
            if config.noise_generator.get("fix_logvar", False):
                s += "-fvr"

    else:
        s = config.exp_id
    return s


class TopCkptManager:
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.ckpt_perf = []  # (ckpt_path, score)

    def save_top_ckpt(self, ckpt_idx, score):
        score = float(score)
        if len(self.ckpt_perf) < self.top_n:
            self.ckpt_perf.append((ckpt_idx, score))
            return None

        self.ckpt_perf.sort(key=lambda x: x[1], reverse=True)

        if score < self.ckpt_perf[-1][1]:
            return ckpt_idx
        else:
            return_item = self.ckpt_perf.pop()
            self.ckpt_perf.append((ckpt_idx, score))
            return return_item[0]


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.gpu_id
        seed_everything(config.seed)

        exp_id = get_exp_id(config)
        self.save_dir = f"{config.training.save_dir}/{config.name}/{exp_id}"

        if os.path.exists(self.save_dir):
            if "test" in exp_id or config.overwrite:
                # Preserve stats.json across overwrites (expensive to recompute)
                stats_backup = None
                stats_file = os.path.join(self.save_dir, "stats.json")
                if os.path.exists(stats_file):
                    import json
                    with open(stats_file) as _f:
                        stats_backup = _f.read()
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
                if stats_backup is not None:
                    with open(stats_file, "w") as _f:
                        _f.write(stats_backup)
            else:
                overwrite = input(
                    f"Save directory {self.save_dir} already exists, overwrite? (y/n)"
                )
                if overwrite != "y":
                    raise ValueError(
                        f"Save directory {self.save_dir} already exists, please delete it or use a different save directory"
                    )
        else:
            os.makedirs(self.save_dir)

        # save config with omegaconf
        with open(
            f"{self.save_dir}/config.yaml", "w"
        ) as f:  # save config to the save_dir
            omegaconf.OmegaConf.save(config, f)

        self.model = hydra.utils.instantiate(config.model)

        if hasattr(self.model, "get_noise_generator"):
            self.model.get_noise_generator(config.noise_generator)
        if config.training.pretrain_ckpt is not None:
            print(f"Loading checkpoint from {config.training.pretrain_ckpt}")
            self.model.load_state_dict(
                torch.load(config.training.pretrain_ckpt, map_location="cpu"),
                strict=False,
            )

        self.model.to(self.device)

        self.use_ema = self.config.training.use_ema
        if self.use_ema:
            self.ema = EMAModel(**config.ema, parameters=self.model.parameters())
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            #self.ema_model = torch.compile(self.ema_model, mode="reduce-overhead")
        else:
            self.ema_model = self.model
            # self.ema_model = torch.compile(self.ema_model, mode="reduce-overhead")
        # Optimizers
        decay_params = [p for p in self.model.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in self.model.parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": 1e-6},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(params=optim_groups, **config.optimizer)

        # Copy pre-computed stats to save_dir so the dataset finds them there
        _src_stats = os.path.join(config.task.dataset.stats_path, "stats.json")
        _dst_stats = os.path.join(self.save_dir, "stats.json")
        if os.path.exists(_src_stats) and not os.path.exists(_dst_stats):
            shutil.copy2(_src_stats, _dst_stats)

        config.task.dataset.recompute_stats = False
        config.task.dataset.stats_path = self.save_dir
        self.dataset = hydra.utils.instantiate(config.task.dataset)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=InfiniteRandomSampler(self.dataset, seed=config.seed),
            **config.dataloader,
        )
        self.lr_scheduler = hydra.utils.instantiate(
            config.training.lr_scheduler, optimizer=self.optimizer
        )
        self.max_training_steps = int(config.training.lr_scheduler.num_training_steps)

        wandb.init(**config.wandb, name=exp_id, tags=exp_id.split("_"))
        self.global_step = 0
        self.ckpt_manager = TopCkptManager(top_n=self.config.training.top_n_ckpt)

        if config.training.resume:
            print(f"Resuming training from {config.training.resume_ckpt}")
            ckpt = torch.load(config.training.resume_ckpt, map_location="cpu")
            self.model.load_state_dict(ckpt["model"], strict=True)
            if "optimizer" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            if "lr_scheduler" in ckpt:
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            if "ckpt_manager" in ckpt:
                self.ckpt_manager.ckpt_perf = ckpt["ckpt_manager"]
            self.global_step = ckpt["global_step"]
            print(f"Resumed training from global step {self.global_step}")

    def run(self):
        self.model.train()

        dataloader_iter = iter(self.dataloader)
        pbar = tqdm(
            total=self.max_training_steps, desc="Training", initial=self.global_step
        )

        import time as _time
        while self.global_step < self.max_training_steps:
            _t_data = _time.time()
            batch = next(dataloader_iter)
            _data_time = _time.time() - _t_data

            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
            loss, train_info = self.model.compute_loss(batch)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0
            ).item()

            # step optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            # update ema
            if self.use_ema:
                self.ema.step(self.model.parameters())

            self.global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", data_s=f"{_data_time:.2f}")

            if self.global_step % self.config.training.log_every_n_steps == 0:
                log_dict = {
                    "train_loss": loss.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "data_time": _data_time,
                }
                log_dict.update(self.process_train_info_for_wandb(train_info))
                wandb.log(log_dict, step=self.global_step)

            if self.global_step % self.config.training.eval_every_n_steps == 0:
                if self.use_ema:
                    self.ema.copy_to(self.ema_model.parameters())
                
                # Save checkpoint regularly
                self.save_ckpt(loss.item())  # Using current loss as performance metric
                
                # Log checkpoint save
                with open(f"{self.save_dir}/ckpt_perf.txt", "a+") as f:
                    f.write(f"{self.global_step} {loss.item()}\n")

        # load down the best ckpts
        with open(f"{self.save_dir}/ckpt_perf.txt", "a+") as f:
            f.write("\n----- best ckpts -----\n")
            for ckpt_idx, ckpt_perf in self.ckpt_manager.ckpt_perf:
                f.write(f"{ckpt_idx} {ckpt_perf}\n")
            f.write(
                f"AVG: {np.mean([perf for _, perf in self.ckpt_manager.ckpt_perf])}\n"
            )

    def save_ckpt(self, ckpt_perf):
        ckpt_dict = {
            "model": remove_prefix(self.ema_model.state_dict()),
            # "optimizer": self.optimizer.state_dict(),
            # "lr_scheduler": self.lr_scheduler.state_dict(),
            # "ema": self.ema.state_dict(),
            "global_step": self.global_step,
            "ckpt_manager": self.ckpt_manager.ckpt_perf,
        }
        torch.save(ckpt_dict, f"{self.save_dir}/ckpt_{self.global_step}.pth")
        print("save ckpt", self.global_step, "Performance:", ckpt_perf)

    def process_train_info_for_wandb(self, train_info):
        # transform some tensor, numpy into scalar
        return train_info

    def process_eval_info_for_wandb(self, eval_info):
        # transform some tensor, numpy into scalar
        log_dict = {"test/mean_score": eval_info["test/mean_score"]}
        return log_dict
