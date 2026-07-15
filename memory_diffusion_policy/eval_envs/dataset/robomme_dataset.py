"""
RoboMME dataset loader using LeRobotDataset (v2.1) for data loading.

LeRobotDataset handles:
  - Arrow-backed parquet loading (cached at HF_DATASETS_CACHE)
  - delta_timestamps windowing for obs_horizon and action_pred_horizon
  - Image decoding and tensor conversion

This wrapper adds:
  - is_demo filtering via episodes_stats.jsonl (O(n_episodes), no row iteration)
  - CLIP text embeddings per task
  - MinMax normalization via DataTransform
"""

import os
# Direct Arrow cache to NFS (has space) before any datasets/lerobot import
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    "/nfs/turbo/coe-chaijy-unreplicated/RoboMME/hf_datasets_cache",
)

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except (ImportError, ModuleNotFoundError):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from eval_envs.utils.normalize import RunningStats
from eval_envs.utils.normalize import save as save_norm_stats
from eval_envs.utils.normalize import load as load_norm_stats
from eval_envs.utils.transform import DataTransform
from eval_envs.utils.clip_model import ClipTextEmbedder

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path):
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class RoboMMEDataset(torch.utils.data.Dataset):
    """
    Loads RoboMME data via LeRobotDataset (v2.1 parquet format).

    LeRobotDataset with delta_timestamps automatically builds:
      - obs_horizon-step observation windows  (image, state)
      - action_pred_horizon-step action chunks

    Only execution frames (is_demo=False) are included; exec_start_idx
    is read from episodes_stats.jsonl for O(n_episodes) filtering.

    Output keys:
      image:    (obs_horizon, 6, H, W)  float32 in [0, 1] — front+wrist concatenated
      state:    (obs_horizon, 8)        float32 (normalized)
      action:   (action_pred_horizon, 8) float32 (normalized)
      text_emb: (obs_horizon, 512)      float32 CLIP embedding (if embed_text=True)
    """

    IMAGE_KEYS = ["image", "wrist_image"]
    STATE_KEY = "state"
    ACTION_KEY = "actions"

    def __init__(
        self,
        dataset_root: str,
        obs_horizon: int = 1,
        action_exec_horizon: int = 8,
        action_pred_horizon: int = 16,
        stats_path: str = "data/robomme",
        recompute_stats: bool = False,
        use_delta_action: bool = False,
        mask: Optional[Sequence[bool]] = None,
        norm_type: str = "minmax",
        embed_text: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ):
        self.dataset_root = Path(dataset_root)
        self.obs_horizon = obs_horizon
        self.action_exec_horizon = action_exec_horizon
        self.action_pred_horizon = action_pred_horizon
        self.embed_text = embed_text

        meta_dir = self.dataset_root / "meta"
        info = json.load(open(meta_dir / "info.json"))
        fps = info["fps"]
        self._data_path_template = info["data_path"]
        self._chunks_size = info["chunks_size"]

        # ---- Build delta_timestamps for LeRobot windowing ----
        obs_ts = [-(obs_horizon - 1 - i) / fps for i in range(obs_horizon)]
        act_ts = [i / fps for i in range(action_pred_horizon)]
        delta_timestamps = {
            self.STATE_KEY: obs_ts,
            self.ACTION_KEY: act_ts,
            **{k: obs_ts for k in self.IMAGE_KEYS},
        }

        # ---- Load LeRobotDataset (Arrow-cached, handles all windowing) ----
        print(f"[RoboMMEDataset] Loading LeRobotDataset from {dataset_root} ...")
        print(f"[RoboMMEDataset] Arrow cache → {os.environ['HF_DATASETS_CACHE']}")
        self.lerobot_dataset = LeRobotDataset(
            repo_id="robomme",
            root=str(self.dataset_root),
            delta_timestamps=delta_timestamps,
        )
        hf_ds = self.lerobot_dataset.hf_dataset
        print(f"[RoboMMEDataset] Loaded {len(hf_ds)} total frames.")

        # ---- Load task texts ----
        self.tasks: Dict[int, str] = {
            t["task_index"]: t["task"] for t in _read_jsonl(meta_dir / "tasks.jsonl")
        }

        # ---- Build episode_index -> task_index mapping (O(n_episodes)) ----
        self.episode_to_task: Dict[int, int] = {}
        ep_data_idx = self.lerobot_dataset.episode_data_index
        for i, ep_meta in enumerate(self.lerobot_dataset.meta.episodes.values()):
            first_row = ep_data_idx["from"][i].item()
            row = hf_ds[first_row]
            ti = row["task_index"]
            if isinstance(ti, torch.Tensor):
                ti = ti.item()
            self.episode_to_task[ep_meta["episode_index"]] = int(ti)

        # ---- Build valid (exec-only) indices from episodes_stats.jsonl (O(n_episodes)) ----
        self._valid_indices: List[int] = self._build_valid_indices(meta_dir)
        total_exec = len(self._valid_indices)
        total_all = len(hf_ds)
        print(f"[RoboMMEDataset] {total_exec} execution samples "
              f"(filtered {total_all - total_exec} demo frames) "
              f"from {len(self.lerobot_dataset.meta.episodes)} episodes.")

        # ---- CLIP text embeddings per task_index ----
        self._task_embeds: Dict[int, np.ndarray] = {}
        if self.embed_text:
            print(f"[RoboMMEDataset] Pre-computing CLIP embeddings for {len(self.tasks)} tasks...")
            clip = ClipTextEmbedder(device="cuda", model_name=clip_model_name)
            task_ids = sorted(self.tasks.keys())
            embeds = clip.embed_texts([self.tasks[tid] for tid in task_ids])
            if isinstance(embeds, torch.Tensor):
                embeds = embeds.detach().cpu().numpy()
            for j, tid in enumerate(task_ids):
                self._task_embeds[tid] = embeds[j].astype(np.float32)
            del clip
            print(f"[RoboMMEDataset] Embedded {len(self._task_embeds)} tasks.")

        # ---- Normalization stats ----
        stats_file = Path(stats_path) / "stats.json"
        if recompute_stats or not stats_file.exists():
            if not recompute_stats:
                print(f"[RoboMMEDataset] Stats not found at {stats_file}, computing now...")
            self.stats = self._compute_stats(meta_dir)
            save_norm_stats(stats_path, self.stats, filename="stats.json")
        else:
            self.stats = load_norm_stats(stats_path, filename="stats.json")

        self.data_transform = DataTransform(
            norm_stats=self.stats,
            norm_type=norm_type,
            mask=mask,
            use_delta_action=use_delta_action,
        )

    # ------------------------------------------------------------------ #
    # Index building                                                        #
    # ------------------------------------------------------------------ #

    def _build_valid_indices(self, meta_dir: Path) -> List[int]:
        """Map exec frames to flat hf_dataset indices using episodes_stats.jsonl."""
        ep_exec_start: Dict[int, int] = {
            s["episode_index"]: int(s["stats"]["exec_start_idx"]["min"][0])
            for s in _read_jsonl(meta_dir / "episodes_stats.jsonl")
        }
        ep_data_idx = self.lerobot_dataset.episode_data_index
        valid: List[int] = []
        for i, ep_meta in enumerate(self.lerobot_dataset.meta.episodes.values()):
            ep_global_start = ep_data_idx["from"][i].item()
            ep_global_end = ep_data_idx["to"][i].item()
            exec_start = ep_exec_start.get(ep_meta["episode_index"], 0)
            valid.extend(range(ep_global_start + exec_start, ep_global_end))
        return valid

    # ------------------------------------------------------------------ #
    # Stats (reads only state+actions columns — no image decode)           #
    # ------------------------------------------------------------------ #

    def _compute_stats(self, meta_dir: Path) -> dict:
        state_stats = RunningStats()
        action_stats = RunningStats()
        episodes_meta = list(_read_jsonl(meta_dir / "episodes.jsonl"))
        ep_exec_start: Dict[int, int] = {
            s["episode_index"]: int(s["stats"]["exec_start_idx"]["min"][0])
            for s in _read_jsonl(meta_dir / "episodes_stats.jsonl")
        }

        print(f"[RoboMMEDataset] Computing stats over {len(episodes_meta)} episodes...")
        for ep in tqdm(episodes_meta, desc="Computing stats"):
            chunk = ep["episode_index"] // self._chunks_size
            path = self.dataset_root / self._data_path_template.format(
                episode_chunk=chunk, episode_index=ep["episode_index"]
            )
            df = pd.read_parquet(path, columns=[self.STATE_KEY, self.ACTION_KEY])
            exec_start = ep_exec_start.get(ep["episode_index"], 0)
            df_exec = df.iloc[exec_start:]
            state_stats.update(np.stack(df_exec[self.STATE_KEY].tolist()).astype(np.float32))
            action_stats.update(np.stack(df_exec[self.ACTION_KEY].tolist()).astype(np.float32))

        print("[RoboMMEDataset] Stats done.")
        return {
            "state": state_stats.get_statistics(),
            "action": action_stats.get_statistics(),
        }

    # ------------------------------------------------------------------ #
    # Dataset interface                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        real_idx = self._valid_indices[idx]
        x = self.lerobot_dataset[real_idx]

        # Concatenate front + wrist: (T, 3, H, W) cat → (T, 6, H, W)
        image = torch.cat([x[k] for k in self.IMAGE_KEYS], dim=1)
        state = x[self.STATE_KEY]    # (T, 8)
        action = x[self.ACTION_KEY]  # (pred_horizon, 8)

        ep_idx = x["episode_index"]
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        task_idx = self.episode_to_task.get(int(ep_idx), 0)

        out = {
            "image":  image.numpy().astype(np.float32),   # (T, 6, H, W)
            "state":  state.numpy().astype(np.float32),   # (T, 8)
            "action": action.numpy().astype(np.float32),  # (pred_horizon, 8)
        }
        if self.embed_text:
            embed = self._task_embeds.get(task_idx, np.zeros(512, dtype=np.float32))
            out["text_emb"] = np.stack([embed] * self.obs_horizon, axis=0)  # (T, 512)

        return self.data_transform.transform_in(out)
