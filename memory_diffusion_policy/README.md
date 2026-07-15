# Diffusion Policy (DP) on RoboMME

This repository contains the Diffusion Policy (DP) baseline trained on the RoboMME real-robot dataset.

## Installation

```bash
conda create -n robomme_dp python=3.11
conda env update -f environment.yml --prune
pip install -e .
# Select PyTorch matching your CUDA version:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# For serve_dp.py:
pip install websockets msgpack msgpack_numpy
```

## Repository Structure

```
DP/
├── eval_envs/
│   ├── config/
│   │   ├── base.yaml                  # base training hyperparameters
│   │   ├── train_dp_unet.yaml         # main training config
│   │   ├── model/dp_unet.yaml         # DP-UNet model config
│   │   └── task/robomme.yaml          # RoboMME dataset config
│   ├── dataset/
│   │   └── robomme_dataset.py         # RoboMME parquet dataset loader
│   ├── model/
│   │   ├── base.py
│   │   ├── dp_unet.py                 # Diffusion Policy UNet
│   │   └── nn_modules.py
│   ├── trainer/
│   │   └── trainer.py                 # Training loop
│   └── utils/
│       ├── clip_model.py
│       ├── normalize.py
│       ├── transform.py
│       └── ...
├── scripts/
│   └── train.py                       # Training entry point
└── serve_dp.py                        # Websocket policy server for eval
```

## Dataset

Set `task.dataset.dataset_root` to the path where you downloaded the RoboMME dataset (LeRobot v2.0 parquet format).

LeRobot v2.0 parquet format — one parquet file per episode, single-step actions (not pre-chunked).

```
robomme_data_lerobot/
    meta/
        info.json
        episodes.jsonl    # {episode_index, tasks, length}
        tasks.jsonl       # {task_index, task}
    data/
        chunk-000/
            episode_000000.parquet
            ...
```

Each parquet row:

| Column | Shape | Description |
|--------|-------|-------------|
| `image` | `(256, 256, 3)` PNG bytes | Front camera |
| `wrist_image` | `(256, 256, 3)` PNG bytes | Wrist camera |
| `state` | `(8,)` float32 | Robot joint state |
| `actions` | `(8,)` float32 | Single-step action |
| `is_demo` | bool | True for demo frames, False for execution frames |
| `exec_start_idx` | int | Frame index where execution starts within episode |

**1,600 episodes, 116 tasks, 768,897 total frames.** Each episode contains a demo portion followed by an execution portion. Only execution frames (`is_demo=False`) are used for training.

## Training

```bash
cd /home/haoranwh/repo/DP
conda activate robomme_dp

PYTHONPATH=$(pwd) python scripts/train.py --config-name=train_dp_unet gpu_id=0
```

Checkpoints are saved to `runs/train_dp_unet/<exp_id>/` (e.g. `dp_bs128_obs1_exe8_pre16_mmnorm`).

### Normalization Stats

On first run the dataset computes normalization stats over all 1,600 episodes and saves them to `<run_dir>/stats.json` (cached after that, only recomputed if `recompute_stats=true`).

### Key Hyperparameters

Config: `eval_envs/config/train_dp_unet.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obs_horizon` | 1 | Observation history length |
| `action_exec_horizon` | 8 | Actions executed per inference step |
| `action_pred_horizon` | 16 | Actions predicted per diffusion pass |
| `dataloader.batch_size` | 128 | Training batch size |
| `norm_type` | minmax | Normalization (`minmax`, `standard`, `quantile`) |
| `include_text` | true | CLIP language conditioning |

Override from command line:

```bash
PYTHONPATH=$(pwd) python scripts/train.py --config-name=train_dp_unet \
    gpu_id=1 obs_horizon=2 dataloader.batch_size=64
```

## Pre-trained Checkpoint

An early checkpoint (`obs_horizon=1`, 200k steps) is available on HuggingFace:
[oldTOM/RoboMME_DP](https://huggingface.co/oldTOM/RoboMME_DP)

The baseline DP model trained with `obs_horizon=2`, `action_exec_horizon=8`, and `action_pred_horizon=16` (config: `train_dp_unet_obs2.yaml`) is coming soon.

## Evaluation (Websocket Server)

**Terminal 1 — start the DP policy server:**

```bash
conda activate robomme_dp
cd /home/haoranwh/repo/DP

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python serve_dp.py \
    --ckpt_path runs/train_dp_unet/dp_bs128_obs1_exe8_pre16_mmnorm/ckpt_***.pth \
    --port 8011
```

**Terminal 2 — run the RoboMME evaluator:**

```bash
conda activate robomme
cd <robomme_policy_learning>

CUDA_VISIBLE_DEVICES=1 python examples/robomme/eval.py \
    --args.host 0.0.0.0 \
    --args.port 8011 \
    --args.policy_name dp_robomme \
    --args.model_ckpt_id ***
```

