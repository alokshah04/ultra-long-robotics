from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import gymnasium as gym

from tests._shared.repo_paths import ensure_src_on_path

ensure_src_on_path(__file__)

from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout  # noqa: E402
from robomme.robomme_env import *  # noqa: F401,F403,E402
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError  # noqa: E402
from robomme.robomme_env.utils.planner_fail_safe import (  # noqa: E402
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)


DATASET_SCREW_MAX_ATTEMPTS = 3
DATASET_RRT_MAX_ATTEMPTS = 3
MAX_SEED_ATTEMPTS = 30


@dataclass(frozen=True)
class DatasetCase:
    env_id: str
    episode: int
    base_seed: int
    difficulty: Optional[str]
    save_video: bool
    mode_tag: str

    def cache_key(self) -> str:
        difficulty = self.difficulty if self.difficulty else "none"
        return (
            f"{self.env_id}_ep{self.episode}_{difficulty}_"
            f"{self.base_seed}_{int(self.save_video)}_{self.mode_tag}"
        )


@dataclass(frozen=True)
class GeneratedDataset:
    case: DatasetCase
    work_dir: Path
    raw_h5_path: Path
    resolver_dataset_dir: Path
    resolver_h5_path: Path
    used_seed: int


def _tensor_to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _patch_planner_screw_to_rrt(planner) -> None:
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args, **kwargs):
        for _ in range(DATASET_SCREW_MAX_ATTEMPTS):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result

        for _ in range(DATASET_RRT_MAX_ATTEMPTS):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        return -1

    planner.move_to_pose_with_screw = _move_screw_then_rrt


def _run_one_episode(
    case: DatasetCase,
    seed: int,
    output_dir: Path,
) -> bool:
    env_kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
        difficulty=case.difficulty,
    )
    if case.episode <= 5:
        env_kwargs["robomme_failure_recovery"] = True
        env_kwargs["robomme_failure_recovery_mode"] = "z" if case.episode <= 2 else "xy"

    env = gym.make(case.env_id, **env_kwargs)
    env = RobommeRecordWrapper(
        env,
        dataset=str(output_dir),
        env_id=case.env_id,
        episode=case.episode,
        seed=seed,
        save_video=case.save_video,
    )

    episode_successful = False
    try:
        env.reset()
        is_stick = case.env_id in ("PatternLock", "RouteStick")
        if is_stick:
            planner = FailAwarePandaStickMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.3,
            )
        else:
            planner = FailAwarePandaArmMotionPlanningSolver(
                env,
                debug=False,
                vis=False,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )

        _patch_planner_screw_to_rrt(planner)

        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        for task_entry in tasks:
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                continue
            env.unwrapped.evaluate(solve_complete_eval=True)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    env.unwrapped.failureflag = torch.tensor([True])
                    env.unwrapped.successflag = torch.tensor([False])
                    env.unwrapped.current_task_failure = True
            except ScrewPlanFailure:
                screw_failed = True
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            except FailsafeTimeout:
                break

            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            fail_flag = evaluation.get("fail", False)
            success_flag = evaluation.get("success", False)

            if _tensor_to_bool(success_flag):
                episode_successful = True
                break
            if screw_failed or _tensor_to_bool(fail_flag):
                break
        else:
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            episode_successful = _tensor_to_bool(evaluation.get("success", False))

        episode_successful = episode_successful or _tensor_to_bool(
            getattr(env, "episode_success", False)
        )
    except SceneGenerationError:
        episode_successful = False
    finally:
        try:
            env.close()
        except Exception:
            pass

    return episode_successful


def _run_episode_with_retry(case: DatasetCase, output_dir: Path) -> tuple[Path, int]:
    for attempt in range(MAX_SEED_ATTEMPTS):
        seed = case.base_seed + attempt
        try:
            success = _run_one_episode(case=case, seed=seed, output_dir=output_dir)
        except Exception:
            continue
        if not success:
            continue

        h5_path = output_dir / "hdf5_files" / f"{case.env_id}_ep{case.episode}_seed{seed}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing expected HDF5: {h5_path}")
        return h5_path, seed
    raise RuntimeError(
        f"[{case.env_id}] Failed to generate successful record in {MAX_SEED_ATTEMPTS} attempts."
    )


def _write_meta(meta_path: Path, payload: dict) -> None:
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def generate_dataset_case(case: DatasetCase, cache_root: Path) -> GeneratedDataset:
    case_dir = cache_root / case.cache_key()
    work_dir = case_dir / "work"
    resolver_dataset_dir = case_dir / "resolver_dataset"
    resolver_h5_path = resolver_dataset_dir / f"record_dataset_{case.env_id}.h5"
    meta_path = case_dir / "meta.json"

    if meta_path.exists():
        meta = _read_meta(meta_path)
        raw_h5_path = Path(meta["raw_h5_path"])
        if raw_h5_path.exists() and resolver_h5_path.exists():
            return GeneratedDataset(
                case=case,
                work_dir=work_dir,
                raw_h5_path=raw_h5_path,
                resolver_dataset_dir=resolver_dataset_dir,
                resolver_h5_path=resolver_h5_path,
                used_seed=int(meta["used_seed"]),
            )

    case_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    resolver_dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_h5_path, used_seed = _run_episode_with_retry(case=case, output_dir=work_dir)
    shutil.copy2(raw_h5_path, resolver_h5_path)

    payload = {
        "case": asdict(case),
        "used_seed": used_seed,
        "raw_h5_path": str(raw_h5_path),
        "resolver_h5_path": str(resolver_h5_path),
    }
    _write_meta(meta_path, payload)

    return GeneratedDataset(
        case=case,
        work_dir=work_dir,
        raw_h5_path=raw_h5_path,
        resolver_dataset_dir=resolver_dataset_dir,
        resolver_h5_path=resolver_h5_path,
        used_seed=used_seed,
    )


class DatasetFactoryCache:
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self._memo: dict[str, GeneratedDataset] = {}

    def get(self, case: DatasetCase) -> GeneratedDataset:
        key = case.cache_key()
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        generated = generate_dataset_case(case, self.cache_root)
        self._memo[key] = generated
        return generated


DatasetFactory = Callable[[DatasetCase], GeneratedDataset]

