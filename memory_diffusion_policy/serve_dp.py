"""
Diffusion Policy (DP) websocket policy server for robomme eval.py.

Usage (in dp_robomme conda env):
    cd /home/haoranwh/repo/DP
    CUDA_VISIBLE_DEVICES=0 \\
    python serve_dp.py \\
        --ckpt_path runs/train_dp_unet_robomme/dp_bs128_obs1_exe8_pre16_mmnorm/ckpt_300000.pth \\
        --port 8011

Then in another terminal (robomme env, from robomme_policy_learning/):
    conda activate robomme
    CUDA_VISIBLE_DEVICES=1 \\
    python examples/robomme/eval.py \\
        --args.host 0.0.0.0 \\
        --args.port 8011 \\
        --args.policy_name dp_robomme \\
        --args.model_ckpt_id 300000
"""

import argparse
import asyncio
import functools
import http
import logging
import os

import msgpack
import numpy as np
import torch
import hydra
import omegaconf

from eval_envs.utils.normalize import load as load_norm_stats
from eval_envs.utils.transform import DataTransform
from eval_envs.utils.distributed_util import remove_prefix
from eval_envs.trainer.trainer import seed_everything
from eval_envs.utils.clip_model import ClipTextEmbedder

import websockets.asyncio.server as _server
import websockets.exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── msgpack codec (same as openpi-client / serve_mtil.py) ──────────────────

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj

def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj

_Packer  = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ── CLIP text embedder ──────────────────────────────────────────────────────
# Loaded eagerly at startup (blocking); must not be called inside async handlers.


# ── image helpers ───────────────────────────────────────────────────────────

def _to_chw_float01(img: np.ndarray) -> np.ndarray:
    """HWC uint8 or float → CHW float32 [0, 1]."""
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img.shape}")
    if img.shape[-1] == 3 and img.shape[0] != 3:
        img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


# ── model loading ────────────────────────────────────────────────────────────

def load_dp_model(ckpt_path: str, device: str, norm_stats_path: str = ""):
    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    config = omegaconf.OmegaConf.load(config_path)
    seed_everything(config.get("seed", 42))

    logger.info(f"Loading model from {ckpt_path}")
    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(remove_prefix(ckpt["model"]), strict=True)
    model.to(device).eval()

    stats_dir = norm_stats_path if norm_stats_path else ckpt_dir
    norm_stats = load_norm_stats(stats_dir, filename="stats.json")
    data_transform = DataTransform(
        norm_stats,
        norm_type=config.norm_type,
        mask=config.task.dataset.mask,
        use_delta_action=config.use_delta_action,
    )
    data_transform.training = False

    obs_horizon = int(config.get("obs_horizon", 1))
    action_exec_horizon = int(config.get("action_exec_horizon", 8))
    include_text = bool(getattr(config.model, "include_text", False))

    clip_embedder = None
    if include_text:
        logger.info("Loading CLIP: openai/clip-vit-base-patch32")
        clip_embedder = ClipTextEmbedder(device=device)

    logger.info(
        f"obs_horizon={obs_horizon}, action_exec_horizon={action_exec_horizon}, "
        f"include_text={include_text}"
    )
    return model, data_transform, obs_horizon, action_exec_horizon, include_text, clip_embedder


# ── per-connection DP state ──────────────────────────────────────────────────

class DPPolicyState:
    def __init__(self, model, data_transform: DataTransform,
                 obs_horizon: int, action_exec_horizon: int,
                 include_text: bool, device: str,
                 clip_embedder: ClipTextEmbedder | None = None):
        self.model = model
        self.data_transform = data_transform
        self.obs_horizon = obs_horizon
        self.action_exec_horizon = action_exec_horizon
        self.include_text = include_text
        self.device = device
        self.clip_embedder = clip_embedder
        self._clip_cache: dict = {}

        # Rolling history buffers (length obs_horizon)
        self._img_buf: list = []    # each: (C, H, W) float32 [0,1]
        self._state_buf: list = []  # each: (8,) float32
        self._text_buf: list = []   # each: (512,) float32

    def reset(self):
        self._img_buf.clear()
        self._state_buf.clear()
        self._text_buf.clear()

    def _push_obs(self, img_chw: np.ndarray, state: np.ndarray, text_emb: np.ndarray):
        self._img_buf.append(img_chw)
        self._state_buf.append(state)
        self._text_buf.append(text_emb)
        if len(self._img_buf) > self.obs_horizon:
            self._img_buf.pop(0)
            self._state_buf.pop(0)
            self._text_buf.pop(0)

    def _get_stacked(self):
        n = len(self._img_buf)
        if n < self.obs_horizon:
            # Pad by repeating the first entry
            pad = self.obs_horizon - n
            imgs   = np.stack([self._img_buf[0]] * pad + self._img_buf,   axis=0)  # (T,C,H,W)
            states = np.stack([self._state_buf[0]] * pad + self._state_buf, axis=0) # (T,8)
            texts  = np.stack([self._text_buf[0]] * pad + self._text_buf,  axis=0)  # (T,512)
        else:
            imgs   = np.stack(self._img_buf[-self.obs_horizon:],   axis=0)
            states = np.stack(self._state_buf[-self.obs_horizon:], axis=0)
            texts  = np.stack(self._text_buf[-self.obs_horizon:],  axis=0)
        return imgs, states, texts

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        # Unpack observation
        img_raw   = obs["observation/image"]        # (H,W,3) or (3,H,W)
        wrist_raw = obs["observation/wrist_image"]  # (H,W,3) or (3,H,W)
        state_raw = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)  # (8,)
        prompt = obs.get("prompt", "")
        if isinstance(prompt, bytes):
            prompt = prompt.decode()

        # Process images
        img_chw   = _to_chw_float01(img_raw)    # (3,H,W)
        wrist_chw = _to_chw_float01(wrist_raw)  # (3,H,W)
        fused     = np.concatenate([img_chw, wrist_chw], axis=0)  # (6,H,W)

        # CLIP text embedding (cache per connection)
        if self.include_text:
            if prompt not in self._clip_cache:
                emb = self.clip_embedder.embed_texts([prompt])  # (1,512) CPU tensor
                self._clip_cache[prompt] = emb[0].float().numpy()
            text_emb = self._clip_cache[prompt]  # (512,)
        else:
            text_emb = np.zeros(512, dtype=np.float32)

        self._push_obs(fused, state_raw, text_emb)
        imgs, states, texts = self._get_stacked()  # (T,*) arrays

        # Build batch
        batch_np = {
            "image": imgs[None, ...].astype(np.float32),    # (1,T,6,H,W)
            "state": states[None, ...].astype(np.float32),  # (1,T,8)
        }
        if self.include_text:
            batch_np["text_emb"] = texts[None, ...].astype(np.float32)  # (1,T,512)

        batch_np = self.data_transform.transform_in(batch_np)

        obs_dict = {
            "image": torch.from_numpy(batch_np["image"]).float().to(self.device),
            "state": torch.from_numpy(batch_np["state"]).float().to(self.device),
        }
        if self.include_text:
            obs_dict["text_emb"] = torch.from_numpy(batch_np["text_emb"]).float().to(self.device)

        out_dict, _ = self.model.predict_action(obs_dict, policy_info={})

        act_norm = out_dict["action"].detach().cpu().numpy()  # (1,K,8) or (K,8)
        act_norm_dict = self.data_transform.transform_out({"action": act_norm})
        act = np.asarray(act_norm_dict["action"], dtype=np.float32)

        if act.ndim == 3:
            act = act[0]   # (K, 8)
        elif act.ndim == 1:
            act = act[None, :]

        actions = act[:self.action_exec_horizon].astype(np.float32)  # (K, 8)
        return {"actions": actions}

    def add_buffer(self, obs: dict):
        """Process a pre-trajectory frame (init obs warmup, no action returned)."""
        self.infer(obs)  # updates rolling buffer, discards output


# ── websocket server ─────────────────────────────────────────────────────────

class DPPolicyServer:
    def __init__(self, model, data_transform, obs_horizon, action_exec_horizon,
                 include_text, device, clip_embedder, host="0.0.0.0", port=8011):
        self.model = model
        self.data_transform = data_transform
        self.obs_horizon = obs_horizon
        self.action_exec_horizon = action_exec_horizon
        self.include_text = include_text
        self.device = device
        self.clip_embedder = clip_embedder
        self.host = host
        self.port = port

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"DP policy server listening on {self.host}:{self.port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address}")
        packer = _Packer()

        # Send empty metadata (expected by MMEVLAWebsocketClientPolicy)
        await websocket.send(packer.pack({}))

        state = DPPolicyState(
            self.model, self.data_transform,
            self.obs_horizon, self.action_exec_horizon,
            self.include_text, self.device,
            self.clip_embedder,
        )

        while True:
            try:
                raw = await websocket.recv()
                obs = _unpackb(raw)

                if obs.get("reset", False):
                    state.reset()
                    await websocket.send(packer.pack({"reset_finished": True}))

                elif obs.get("add_buffer", False):
                    # DP is non-recurrent; pre-trajectory warmup not needed
                    await websocket.send(packer.pack({"add_buffer_finished": True}))

                else:
                    outputs = state.infer(obs)
                    await websocket.send(packer.pack(outputs))

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logger.error(f"Handler error: {type(e).__name__}: {e}", exc_info=True)
                break

    @staticmethod
    async def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Serve a DP checkpoint over websockets.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to DP checkpoint (.pth), config.yaml must be in same directory")
    parser.add_argument("--norm_stats_path", type=str, default="",
                        help="Path to directory containing stats.json (default: same as ckpt_path)")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (default: cuda:0 if available)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, data_transform, obs_horizon, action_exec_horizon, include_text, clip_embedder = load_dp_model(
        args.ckpt_path, device, args.norm_stats_path
    )

    server = DPPolicyServer(
        model, data_transform, obs_horizon, action_exec_horizon,
        include_text, device, clip_embedder, host=args.host, port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
