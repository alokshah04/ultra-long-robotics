import importlib.util
from pathlib import Path

from tests._shared.repo_paths import find_repo_root


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


matcher_mod = _load_module(
    "oracle_action_matcher_position_under_test",
    "src/robomme/robomme_env/utils/oracle_action_matcher.py",
)


class _Pose:
    def __init__(self, p):
        self.p = p


class _Actor:
    def __init__(self, name, pose_p=None, use_get_pose=False):
        self.name = name
        self._pose = _Pose(pose_p) if pose_p is not None else None
        if not use_get_pose and self._pose is not None:
            self.pose = self._pose

    def get_pose(self):
        if self._pose is None:
            raise RuntimeError("missing pose")
        return self._pose


def test_select_target_with_position_nearest_candidate():
    actor_a = _Actor("A", [0.0, 0.0, 0.0])
    actor_b = _Actor("B", [0.8, 0.0, 0.0])
    actor_c = _Actor("C", [3.0, 0.0, 0.0])

    result = matcher_mod.select_target_with_position(
        available=[actor_a, actor_b, actor_c],
        position_like=[1.0, 0.0, 0.0],
    )

    assert result is not None
    assert result["name"] == "B"
    assert result["selection_mode"] == "nearest_position"
    assert abs(float(result["match_distance"]) - 0.2) < 1e-9


def test_select_target_with_position_skips_invalid_candidates():
    actor_invalid = _Actor("invalid", None)
    actor_valid = _Actor("valid", [1.0, 2.0, 3.0], use_get_pose=True)

    result = matcher_mod.select_target_with_position(
        available=[actor_invalid, actor_valid],
        position_like=[1.2, 2.2, 3.1],
    )
    assert result is not None
    assert result["name"] == "valid"


def test_select_target_with_position_returns_none_without_valid_input():
    actor_invalid = _Actor("invalid", None)

    res_invalid_position = matcher_mod.select_target_with_position(
        available=[_Actor("a", [0.0, 0.0, 0.0])],
        position_like=[0.0, 0.0],
    )
    assert res_invalid_position is None

    res_invalid_candidates = matcher_mod.select_target_with_position(
        available=[actor_invalid],
        position_like=[0.0, 0.0, 0.0],
    )
    assert res_invalid_candidates is None


def test_select_target_with_position_tie_breaks_by_first_flattened_candidate():
    actor_first = _Actor("first", [1.0, 0.0, 0.0])
    actor_second = _Actor("second", [-1.0, 0.0, 0.0])

    result = matcher_mod.select_target_with_position(
        available={"left": [actor_first], "right": [actor_second]},
        position_like=[0.0, 0.0, 0.0],
    )

    assert result is not None
    assert result["name"] == "first"
