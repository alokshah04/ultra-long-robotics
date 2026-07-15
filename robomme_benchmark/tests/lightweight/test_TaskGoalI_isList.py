"""
test_TaskGoalIsList.py

Directly create a real Gymnasium environment (wrapped in DemonstrationWrapper),
call env.reset(), and verify that info["task_goal"] is a list and not empty.

Covers all 16 envs.

Run:
    uv run python -m pytest tests/lightweight/test_TaskGoalI_isList.py -v -s
"""

import gymnasium as gym
import pytest
from typing import Literal

from robomme.env_record_wrapper.DemonstrationWrapper import DemonstrationWrapper
from robomme.env_record_wrapper.EndeffectorDemonstrationWrapper import EndeffectorDemonstrationWrapper
from robomme.env_record_wrapper.MultiStepDemonstrationWrapper import MultiStepDemonstrationWrapper
from robomme.env_record_wrapper.OraclePlannerDemonstrationWrapper import OraclePlannerDemonstrationWrapper

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

# ── All 16 env_ids ─────────────────────────────────────────────────────────────
ALL_ENV_IDS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
]

# ── Four ActionSpaceTypes ──────────────────────────────────────────────────────
ACTION_SPACES = ["joint_angle", "ee_pose", "waypoint", "multi_choice"]


def _make_env(env_id: str, action_space: str):
    """Create and return a real environment wrapped with the corresponding Wrapper."""
    env = gym.make(
        env_id,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    # Base Wrapper
    env = DemonstrationWrapper(
        env,
        max_steps_without_demonstration=10002,
        gui_render=False,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )

    # Apply additional Wrapper depending on action_space
    if action_space == "joint_angle":
        pass
    elif action_space == "ee_pose":
        env = EndeffectorDemonstrationWrapper(env, action_repr="rpy")
    elif action_space == "waypoint":
        env = MultiStepDemonstrationWrapper(env, gui_render=False, vis=False)
    elif action_space == "multi_choice":
        env = OraclePlannerDemonstrationWrapper(env, env_id=env_id, gui_render=False)
    else:
        raise ValueError(f"Unsupported action_space: {action_space}")

    return env


@pytest.mark.parametrize(
    "env_id, action_space",
    [(env, action) for env in ALL_ENV_IDS for action in ACTION_SPACES],
)
def test_task_goal_is_list(env_id: str, action_space: str):
    """
    Test four action_spaces consecutively for each env_id:
    1. Create real environment (including corresponding Wrapper)
    2. Call reset()
    3. Assert info["task_goal"] is list and not empty
    """
    print(f"\nTesting [{env_id}] with action_space={action_space!r}")
    env = _make_env(env_id, action_space)
    try:
        _, info = env.reset()
    finally:
        env.close()

    task_goal = info["task_goal"]
    print(f"[{env_id} | {action_space}] task_goal = {task_goal!r}")

    assert isinstance(task_goal, list), (
        f"[{env_id} | {action_space}] info['task_goal'] should be list, actually {type(task_goal).__name__!r}: {task_goal!r}"
    )
    assert len(task_goal) >= 1, (
        f"[{env_id} | {action_space}] info['task_goal'] should not be empty list"
    )
    for i, item in enumerate(task_goal):
        assert isinstance(item, str), (
            f"[{env_id} | {action_space}] task_goal[{i}] should be str, actually {type(item).__name__!r}: {item!r}"
        )
