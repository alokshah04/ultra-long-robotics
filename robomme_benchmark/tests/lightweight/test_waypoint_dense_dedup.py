# -*- coding: utf-8 -*-
"""
test_waypoint_dense_dedup.py
============================
Unit test: Verify that EpisodeDatasetResolver's waypoint read logic
changes from depending on info/is_keyframe to scanning dense waypoint_action and strictly deduplicating adjacent ones.

Run (using uv):
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/lightweight/test_waypoint_dense_dedup.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tests._shared.repo_paths import find_repo_root

_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "robomme" / "env_record_wrapper"))

from episode_dataset_resolver import EpisodeDatasetResolver


def _make_timestep(
    episode_group: h5py.Group,
    timestep_idx: int,
    *,
    waypoint_action: np.ndarray | None = None,
    is_video_demo: bool = False,
    is_keyframe: bool = False,
) -> None:
    ts = episode_group.create_group(f"timestep_{timestep_idx}")
    info = ts.create_group("info")
    info.create_dataset("is_video_demo", data=is_video_demo)
    info.create_dataset("is_keyframe", data=is_keyframe)
    if waypoint_action is not None:
        action = ts.create_group("action")
        action.create_dataset("waypoint_action", data=np.asarray(waypoint_action, dtype=np.float32))


def _build_h5(h5_path: Path, timestep_specs: list[dict[str, Any]]) -> None:
    with h5py.File(h5_path, "w") as h5:
        ep = h5.create_group("episode_0")
        for idx, spec in enumerate(timestep_specs):
            _make_timestep(ep, idx, **spec)


def _assert_waypoint_sequence(h5_path: Path, expected: list[np.ndarray]) -> None:
    resolver = EpisodeDatasetResolver(env_id="Dummy", episode=0, dataset_directory=str(h5_path))
    try:
        assert resolver.get_step("waypoint", -1) is None, "negative step should return None"

        seq = []
        for idx, exp in enumerate(expected):
            got = resolver.get_step("waypoint", idx)
            assert got is not None, f"expected waypoint at step {idx}, got None"
            got_arr = np.asarray(got)
            assert got_arr.shape == (7,), f"step {idx} shape expected (7,), got {got_arr.shape}"
            assert np.array_equal(got_arr, exp), f"step {idx} mismatch: {got_arr} != {exp}"
            seq.append(got_arr)

        assert resolver.get_step("waypoint", len(expected)) is None, "oob step should return None"
        assert len(seq) == len(expected)
    finally:
        resolver.close()


def _case_adjacent_dedup_ignore_keyframe(tmpdir: Path) -> None:
    a = np.array([1, 2, 3, 4, 5, 6, -1], dtype=np.float32)
    b = np.array([10, 20, 30, 40, 50, 60, 1], dtype=np.float32)
    c = np.array([7, 8, 9, -4, -5, -6, -1], dtype=np.float32)

    h5_path = tmpdir / "case1.h5"
    _build_h5(
        h5_path,
        [
            {"waypoint_action": a, "is_keyframe": False},                     # t0 -> A
            {"waypoint_action": a, "is_keyframe": True},                      # t1 -> A (misleading keyframe)
            {"waypoint_action": b, "is_video_demo": True, "is_keyframe": True},  # t2 skipped
            {"waypoint_action": b, "is_keyframe": False},                     # t3 -> B
            {"waypoint_action": b, "is_keyframe": False},                     # t4 -> B
            {"waypoint_action": c, "is_keyframe": False},                     # t5 -> C
            {"waypoint_action": a, "is_keyframe": False},                     # t6 -> A (non-adjacent repeat kept)
        ],
    )

    _assert_waypoint_sequence(h5_path, [a, b, c, a])


def _case_all_keyframe_false(tmpdir: Path) -> None:
    a = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    b = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    h5_path = tmpdir / "case2.h5"
    _build_h5(
        h5_path,
        [
            {"waypoint_action": a, "is_keyframe": False},
            {"waypoint_action": a, "is_keyframe": False},
            {"waypoint_action": b, "is_keyframe": False},
            {"waypoint_action": b, "is_keyframe": False},
        ],
    )
    _assert_waypoint_sequence(h5_path, [a, b])


def _case_missing_waypoint_action_skipped(tmpdir: Path) -> None:
    a = np.array([3, 3, 3, 3, 3, 3, -1], dtype=np.float32)
    b = np.array([4, 4, 4, 4, 4, 4, 1], dtype=np.float32)
    h5_path = tmpdir / "case3.h5"
    _build_h5(
        h5_path,
        [
            {"waypoint_action": a, "is_keyframe": False},
            {"is_keyframe": True},  # missing action/waypoint_action entirely
            {"waypoint_action": b, "is_keyframe": False},
            {"waypoint_action": b, "is_keyframe": False},
        ],
    )
    _assert_waypoint_sequence(h5_path, [a, b])


def _case_non_finite_waypoint_skipped(tmpdir: Path) -> None:
    a = np.array([5, 5, 5, 5, 5, 5, -1], dtype=np.float32)
    b = np.array([6, 6, 6, 6, 6, 6, 1], dtype=np.float32)
    nan_waypoint = np.full(7, np.nan, dtype=np.float32)
    inf_waypoint = np.array([0, 0, 0, 0, 0, 0, np.inf], dtype=np.float32)

    h5_path = tmpdir / "case4.h5"
    _build_h5(
        h5_path,
        [
            {"waypoint_action": a, "is_keyframe": False},
            {"waypoint_action": nan_waypoint, "is_keyframe": False},  # non-finite sentinel, skipped
            {"waypoint_action": inf_waypoint, "is_keyframe": False},  # non-finite invalid value, skipped
            {"waypoint_action": b, "is_keyframe": False},
            {"waypoint_action": b, "is_keyframe": False},
        ],
    )
    _assert_waypoint_sequence(h5_path, [a, b])


def main() -> None:
    print("\n[TEST] EpisodeDatasetResolver waypoint dense dedup")
    with tempfile.TemporaryDirectory(prefix="waypoint_dense_dedup_") as tmp:
        tmpdir = Path(tmp)

        _case_adjacent_dedup_ignore_keyframe(tmpdir)
        print("  case1 ✓ Adjacent deduplication + ignore is_keyframe + skip video_demo")

        _case_all_keyframe_false(tmpdir)
        print("  case2 ✓ Can still extract even if all is_keyframe=False")

        _case_missing_waypoint_action_skipped(tmpdir)
        print("  case3 ✓ Timesteps missing waypoint_action are skipped")

        _case_non_finite_waypoint_skipped(tmpdir)
        print("  case4 ✓ Timesteps with non-finite waypoint_action are skipped")

    print("\nPASS: waypoint dense dedup tests passed")


if __name__ == "__main__":
    main()
