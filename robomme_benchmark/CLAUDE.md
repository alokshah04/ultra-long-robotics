# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RoboMME is a robotic simulation benchmark (built on ManiSkill/SAPIEN physics) for memory-augmented manipulation. It defines 16 tasks across 4 cognitively-motivated suites (Counting/temporal, Permanence/spatial, Reference/object, Imitation/procedural memory), exposes them as Gymnasium envs, and ships tooling to record demonstrations, replay HDF5 datasets, and evaluate policies. `challenge_interface/` is a separate policy-serving protocol used for the RoboMME Challenge (CVPR 2026), largely decoupled from the core benchmark code.

## Commands

Package management is via `uv`; always prefix commands with `uv run` from the repo root.

```bash
# Install
uv sync
uv pip install -e .

# Sanity-check env + action space, writes an mp4 to runs/sample_run_videos
uv run scripts/run_example.py

# Replay a downloaded HDF5 dataset
uv run scripts/dataset_replay.py --h5-data-dir <your_downloaded_data_dir>

# Run all tests
uv run python -m pytest tests/

# Fast, non-physics unit tests only
uv run python -m pytest tests/lightweight/

# Physics/simulation + dataset-alignment tests (slower)
uv run python -m pytest tests/dataset/

# Single file / single test
uv run python -m pytest tests/lightweight/test_TaskGoal.py
uv run python -m pytest tests/lightweight/test_TaskGoal.py::test_binfill_two_colors

# Run by pytest marker (slow, gpu, dataset, lightweight)
uv run python -m pytest -m dataset

# Show print()/stdout during tests
uv run python -m pytest tests/ -s
```

Docker (for headless/GPU rendering environments, see `doc/docker_installation.md`):
```bash
docker build -t robomme:cuda12.8 .
docker run --rm -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video -v "$PWD/runs:/app/runs" robomme:cuda12.8
```

Challenge policy server (only relevant when working in `challenge_interface/`):
```bash
uv sync --group server
uv run python -m challenge_interface.scripts.deploy --port 8001       # participant server
uv run python -m challenge_interface.scripts.phase1_eval --port 8001  # organizer eval client
```

Python 3.11 required. `mani-skill` is pinned to a fork (`YinpeiDai/ManiSkill`, see `pyproject.toml` `[tool.uv.sources]`) — don't assume upstream ManiSkill APIs are unmodified.

## Architecture

### Layered env construction

The core abstraction is a wrapper stack, assembled by `BenchmarkEnvBuilder.make_env_for_episode()` in `src/robomme/env_record_wrapper/episode_config_resolver.py`:

1. **Base task env** — one class per task in `src/robomme/robomme_env/<TaskName>.py` (e.g. `PickXtimes.py` defines `class PickXtimes(BaseEnv)`), registered with Gymnasium and instantiated via `gym.make(env_id, ...)`. All tasks are exported through `src/robomme/robomme_env/__init__.py`.
2. **`DemonstrationWrapper`** (`env_record_wrapper/DemonstrationWrapper.py`, largest file at ~870 lines) — always applied. Normalizes obs/info into the list-based schema (every obs value is a list of frames, not a single item — see `doc/env_format.md`), handles the `include_*` optional-field switches, and manages `info["status"]` (`success`/`fail`/`timeout`/`ongoing`/`error`).
3. **Action-space wrapper** — chosen by `action_space` argument:
   - `joint_angle`: no extra wrapper (raw 7 joints + gripper).
   - `ee_pose`: `EndeffectorDemonstrationWrapper` (IK-based end-effector control).
   - `waypoint`: `MultiStepDemonstrationWrapper` (discrete keyframe execution).
   - `multi_choice`: `OraclePlannerDemonstrationWrapper` (VideoQA-style discrete choice actions, forces front-camera intrinsics/extrinsics on).
4. **`FailAwareWrapper`** — outermost, always applied last; catches physics/IK errors from inner layers so `env.step()` never crashes, surfacing `info["status"] = "error"` instead.

Separately, `RecordWrapper.py` (~1400 lines, the other large file) drives demonstration recording into the HDF5 format described in `doc/h5_data_format.md`, and `episode_dataset_resolver.py` (`EpisodeDatasetResolver`) reads that format back for replay — the two are expected to round-trip exactly (see `tests/dataset/test_record_stick.py` / `test_replay_stick.py`).

### Episode/metadata resolution

Per-episode determinism (seed, difficulty) comes from JSON metadata under `src/robomme/env_metadata/<train|test|val>/record_dataset_<TaskID>_metadata.json`, loaded by `load_episode_metadata` / `get_episode_metadata`. `BenchmarkEnvBuilder.get_episode_num()` derives episode counts from this metadata rather than any hardcoded constant. Train has 100 episodes/task; val/test have 50 each.

### Task-internal structure

Each task file typically composes shared logic from `src/robomme/robomme_env/utils/`: subgoal language generation (`subgoal_language.py`, `task_goal.py`), oracle planners (`subgoal_planner_func.py`, `planner_denseStep.py`, `planner_fail_safe.py`), scene/object spawning (`object_generation.py`), action matching for replay (`oracle_action_matcher.py`), and pixel/world coordinate projection for `multi_choice` grounding (`choice_action_mapping.py`). When touching a task env, check `utils/` first — most cross-task behavior lives there, not duplicated per task.

World coordinate frame (used by `ee_pose`/`waypoint` actions and `eef_state_list`): right-handed, `+x` forward from robot to workspace, `+y` robot's left, `+z` up; table-top is `z=0`, world origin is the table-top center; Panda base is fixed at `(-0.615, 0, 0)`. RPY is extrinsic XYZ Euler, unwrapped (unbounded), not principal values. Full details in `doc/env_format.md`.

### Test architecture

- `tests/lightweight/`: pure logic/branch tests (label matching, pixel projection math, subgoal text generation), no physics — fast, run these first when iterating.
- `tests/dataset/`: exercises the real physics engine and full wrapper stack; expensive dataset generation is memoized for the test session via a hash-based cache (`tests/_shared/dataset_generation.py`, `dataset_factory` fixture in `tests/dataset/conftest.py`) so the same demonstration trajectory isn't regenerated across tests.
- See `tests/README.md` for a per-file description of what each test actually asserts — read it before adding a new test in either directory, since the fixture/caching conventions matter for keeping `dataset/` tests fast.

### Action/observation conventions

- Gripper: `-1` = closed, `1` = open. Actions are absolute, not deltas.
- `obs` values are always lists (supports conditioning-video frames + intermediate observations for discrete actions); use `obs[key][-1]` for the latest frame only.
- `info["status"]` is the canonical episode-outcome signal (`success`, `fail`, `timeout`, `ongoing`, `error`) — callers should check this rather than relying on bare exceptions, since `FailAwareWrapper` converts errors into this field.
