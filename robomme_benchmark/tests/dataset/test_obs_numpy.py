# -*- coding: utf-8 -*-
"""
test_obs_numpy.py
===================
Integration test: Directly call the real environment + unified temporary dataset generated at test runtime,
test the native type conversions inside DemonstrationWrapper._augment_obs_and_info
and whether the output types and shapes of obs/info fields are correct under the four ActionSpaces.

Covered ActionSpaces:
    joint_angle / ee_pose / waypoint / multi_choice

Asserts content:
  1. Returned dtype complies with specifications (e.g. uint8, int16, float32, float64, etc.)
  2. Non-Tensor field types in info meet expectations

Run (must use uv):
    cd /data/hongzefu/robomme_benchmark
    uv run python -m pytest tests/dataset/test_obs_numpy.py -v -s
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pytest

from tests._shared.repo_paths import find_repo_root

# Ensure src path can be found
pytestmark = pytest.mark.dataset

_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.robomme_env import *  # noqa: F401,F403  Register all custom environments
from robomme.robomme_env.utils import *  # noqa: F401,F403
from robomme.env_record_wrapper import BenchmarkEnvBuilder, EpisodeDatasetResolver

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
TEST_ENV_ID = "VideoUnmaskSwap"
TEST_EPISODE = 0
MAX_STEPS_PER_ACTION_SPACE = 3   # Max steps to verify per ActionSpace
MAX_STEPS_ENV = 1000

ActionSpaceType = Literal["joint_angle", "ee_pose", "waypoint", "multi_choice"]

# ──────────────────────────────────────────────────────────────────────────────
# Assertion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _assert_ndarray(val: Any, dtype: np.dtype, tag: str) -> None:
    assert isinstance(val, np.ndarray), (
        f"[{tag}] expected ndarray, got {type(val).__name__}"
    )
    assert val.dtype == dtype, (
        f"[{tag}] expected dtype={dtype}, got {val.dtype}"
    )


def _assert_ndarray_loose(val: Any, tag: str) -> None:
    """Only assert it is an ndarray, do not check specific dtype."""
    assert isinstance(val, np.ndarray), (
        f"[{tag}] expected ndarray, got {type(val).__name__}"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Core assertion: native output type is correct
# ──────────────────────────────────────────────────────────────────────────────

def assert_obs(obs: dict, tag: str) -> None:
    """Assert obs output dtype is correct and shape matches expectation."""
    n = len(obs.get("front_rgb_list", []))
    assert n > 0, f"[{tag}] obs front_rgb_list is empty"

    for i in range(n):
        pfx = f"{tag}[{i}]"

        # ── RGB → uint8 ───────────────────────────────────────────────────
        for key, dtype in (("front_rgb_list", np.uint8), ("wrist_rgb_list", np.uint8)):
            _assert_ndarray(obs[key][i], dtype, f"{pfx} {key}")

        # ── Depth → int16 ─────────────────────────────────────────────────
        for key, dtype in (("front_depth_list", np.int16), ("wrist_depth_list", np.int16)):
            _assert_ndarray(obs[key][i], dtype, f"{pfx} {key}")

        # ── eef_state_list → float64, shape (6,) ─────────────────────────
        eef_state = obs["eef_state_list"][i]
        _assert_ndarray(eef_state, np.float64, f"{pfx} eef_state_list")
        assert eef_state.shape == (6,), (
            f"[{pfx} eef_state_list] expected shape (6,), got {eef_state.shape}"
        )

        # ── joint_state_list → ndarray (shape unchanged) ───────────────────────
        _assert_ndarray_loose(obs["joint_state_list"][i], f"{pfx} joint_state_list")

        # ── gripper_state_list → ndarray (shape unchanged) ─────────────────────
        _assert_ndarray_loose(obs["gripper_state_list"][i], f"{pfx} gripper_state_list")

        # ── camera extrinsics → float32, shape (3,4) ───────────────────────
        for key in ("front_camera_extrinsic_list", "wrist_camera_extrinsic_list"):
            _assert_ndarray(obs[key][i], np.float32, f"{pfx} {key}")
            assert obs[key][i].shape == (3, 4), (
                f"[{pfx} {key}] expected (3, 4), got {obs[key][i].shape}"
            )


def assert_info(info: dict, tag: str) -> None:
    """Assert the dtypes of info output fields are correct."""
    for key in ("front_camera_intrinsic", "wrist_camera_intrinsic"):
        assert key in info, f"[{tag}] info missing key '{key}'"
        _assert_ndarray(info[key], np.float32, f"{tag} info['{key}']")
        assert info[key].shape == (3, 3), (
            f"[{tag} info['{key}']] expected (3, 3), got {info[key].shape}"
        )

    # Non-Tensor field types unchanged
    task_goal = info.get("task_goal")
    assert isinstance(task_goal, (str, list, type(None))), (
        f"[{tag}] info['task_goal'] unexpected type {type(task_goal)}"
    )
    status = info.get("status")
    assert isinstance(status, (str, type(None))), (
        f"[{tag}] info['status'] unexpected type {type(status)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full episode test for a single ActionSpace
# ──────────────────────────────────────────────────────────────────────────────

def _parse_oracle_command(choice_action: Optional[Any]) -> Optional[dict]:
    """Oracle command parsing consistent with dataset_replay—printType.py."""
    if not isinstance(choice_action, dict):
        return None
    choice = choice_action.get("choice")
    if not isinstance(choice, str) or not choice.strip():
        return None
    if "point" not in choice_action:
        return None
    return {"choice": choice_action.get("choice"), "point": choice_action.get("point")}


def run_one_action_space(action_space: ActionSpaceType, dataset_root: str | Path) -> None:
    print(f"\n{'='*60}")
    print(f"[TEST] ActionSpace = {action_space}")
    print(f"{'='*60}")

    # multi_choice uses OraclePlannerDemonstrationWrapper,
    # BenchmarkEnvBuilder directly uses unified action_space naming.

    env_builder = BenchmarkEnvBuilder(
        env_id=TEST_ENV_ID,
        dataset="train",
        action_space=action_space,
        gui_render=False,
    )
    env = env_builder.make_env_for_episode(
        TEST_EPISODE,
        max_steps=MAX_STEPS_ENV,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )

    dataset_resolver = EpisodeDatasetResolver(
        env_id=TEST_ENV_ID,
        episode=TEST_EPISODE,
        dataset_directory=str(dataset_root),
    )

    # ── RESET ──────────────────────────────────────────────────────────────
    obs, info = env.reset()

    reset_tag = f"{TEST_ENV_ID} ep{TEST_EPISODE} RESET [{action_space}]"
    assert_obs(obs, reset_tag)
    assert_info(info, reset_tag)
    print(f"  RESET assertion passed  (obs list len={len(obs['front_rgb_list'])}, dtype ✓)")

    # ── STEP LOOP ──────────────────────────────────────────────────────────
    step = 0
    while step < MAX_STEPS_PER_ACTION_SPACE:
        replay_key = action_space
        action = dataset_resolver.get_step(replay_key, step)
        if action_space == "multi_choice":
            action = _parse_oracle_command(action)
        if action is None:
            print(f"  step {step}: action=None (dataset ended), breaking out")
            break

        obs, reward, terminated, truncated, info = env.step(action)

        step_tag = f"{TEST_ENV_ID} ep{TEST_EPISODE} STEP{step} [{action_space}]"
        assert_obs(obs, step_tag)
        assert_info(info, step_tag)
        print(f"  STEP {step} assertion passed  (obs list len={len(obs['front_rgb_list'])}, dtype ✓)")

        terminated_flag = bool(terminated.item())
        truncated_flag = bool(truncated.item())
        step += 1
        if terminated_flag or truncated_flag:
            print(f"  terminated={terminated_flag} truncated={truncated_flag}, exiting early")
            break

    env.close()
    print(f"  [{action_space}] ✓ All assertions passed (total {step} steps)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SPACES: list[ActionSpaceType] = [
    "joint_angle",
    "ee_pose",
    "waypoint",
    "multi_choice",
]


@pytest.mark.parametrize("action_space", ACTION_SPACES)
def test_obs_numpy_action_space(action_space: ActionSpaceType, video_unmaskswap_train_ep0_dataset) -> None:
    run_one_action_space(action_space, video_unmaskswap_train_ep0_dataset.resolver_dataset_dir)


def main() -> None:
    print("test_obs_numpy main() now relies on pytest fixture-generated dataset.")
    print("Run with: uv run python -m pytest tests/dataset/test_obs_numpy.py -v -s")
    sys.exit(2)


if __name__ == "__main__":
    main()
