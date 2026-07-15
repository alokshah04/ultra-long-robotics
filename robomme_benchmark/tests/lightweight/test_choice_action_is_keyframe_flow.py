# -*- coding: utf-8 -*-
"""
Lightweight test: choice_action new schema + EpisodeDatasetResolver read by info/is_subgoal_boundary.

Run (using uv):
    uv run python tests/lightweight/test_choice_action_is_keyframe_flow.py
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import h5py
import pytest

from tests._shared.repo_paths import find_repo_root

pytestmark = [pytest.mark.lightweight, pytest.mark.gpu]


def _load_module(module_name: str, relative_path: str):
    repo_root = find_repo_root(__file__)
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resolver_mod = _load_module(
    "episode_dataset_resolver_under_test",
    "src/robomme/env_record_wrapper/episode_dataset_resolver.py",
)


def _make_timestep(
    episode_group: h5py.Group,
    timestep_idx: int,
    *,
    choice_action: dict | None = None,
    is_video_demo: bool = False,
    is_subgoal_boundary: bool = False,
) -> None:
    ts = episode_group.create_group(f"timestep_{timestep_idx}")
    action = ts.create_group("action")
    payload = "{}" if choice_action is None else json.dumps(choice_action)
    action.create_dataset(
        "choice_action",
        data=payload,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    info = ts.create_group("info")
    info.create_dataset("is_video_demo", data=is_video_demo)
    info.create_dataset("is_subgoal_boundary", data=is_subgoal_boundary)


def _build_h5(h5_path: Path) -> None:
    with h5py.File(h5_path, "w") as h5:
        ep = h5.create_group("episode_0")

        # Non-subgoal boundary: valid choice must also be ignored
        _make_timestep(
            ep,
            0,
            choice_action={
                "choice": "A",
                "point": [20, 10],
            },
            is_subgoal_boundary=False,
        )
        # Valid subgoal boundary: should be read
        _make_timestep(
            ep,
            1,
            choice_action={
                "choice": "B",
                "point": [34, 12],
            },
            is_subgoal_boundary=True,
        )
        # Subgoal boundary but empty choice: skip
        _make_timestep(
            ep,
            2,
            choice_action={
                "choice": "",
                "point": [30, 20],
            },
            is_subgoal_boundary=True,
        )
        # Video demo subgoal boundary: skip
        _make_timestep(
            ep,
            3,
            choice_action={
                "choice": "C",
                "point": [80, 70],
            },
            is_video_demo=True,
            is_subgoal_boundary=True,
        )
        # Second valid subgoal boundary
        _make_timestep(
            ep,
            4,
            choice_action={
                "choice": "D",
                "point": [11, 90],
            },
            is_subgoal_boundary=True,
        )

        # legacy schema should be ignored under strict mode.
        ts5 = ep.create_group("timestep_5")
        ts5_action = ts5.create_group("action")
        ts5_action.create_dataset(
            "choice_action",
            data=json.dumps({"label": "legacy", "position": [50, 60]}),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        ts5_info = ts5.create_group("info")
        ts5_info.create_dataset("is_video_demo", data=False)
        ts5_info.create_dataset("is_keyframe", data=True)
        ts5_info.create_dataset("is_subgoal_boundary", data=True)
        
        # new boundary but missing point key should also be ignored.
        _make_timestep(
            ep,
            6,
            choice_action={"choice": "MISSING_POINT"},
            is_subgoal_boundary=True,
        )
        # point can be empty list and should still be read as-is.
        _make_timestep(
            ep,
            7,
            choice_action={"choice": "E", "point": []},
            is_subgoal_boundary=True,
        )


def _assert_record_schema_contract(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5:
        ts1 = h5["episode_0"]["timestep_1"]
        raw = ts1["action"]["choice_action"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        assert "serial_number" not in payload, "choice_action should not store serial_number"
        assert payload["choice"] == "B"
        assert payload["point"] == [34, 12]
        assert "position_3d" not in payload
        assert bool(ts1["info"]["is_subgoal_boundary"][()]) is True


def _assert_resolver_reads_by_is_subgoal_boundary(h5_path: Path) -> None:
    resolver = resolver_mod.EpisodeDatasetResolver(
        env_id="Dummy",
        episode=0,
        dataset_directory=str(h5_path),
    )
    try:
        assert resolver.get_step("multi_choice", -1) is None

        command0 = resolver.get_step("multi_choice", 0)
        assert command0 == {"choice": "B", "point": [34, 12]}
        assert "position_3d" not in command0
        assert "serial_number" not in command0

        command1 = resolver.get_step("multi_choice", 1)
        assert command1 == {"choice": "D", "point": [11, 90]}
        assert "position_3d" not in command1
        assert "serial_number" not in command1

        command2 = resolver.get_step("multi_choice", 2)
        assert command2 == {"choice": "E", "point": []}
        assert resolver.get_step("multi_choice", 3) is None
    finally:
        resolver.close()


def test_choice_action_is_keyframe_flow_pytest(tmp_path: Path) -> None:
    h5_path = tmp_path / "choice_action_flow.h5"
    _build_h5(h5_path)
    _assert_record_schema_contract(h5_path)
    _assert_resolver_reads_by_is_subgoal_boundary(h5_path)


def main() -> None:
    print("\n[TEST] choice_action is_subgoal_boundary flow")
    with tempfile.TemporaryDirectory(prefix="choice_action_is_keyframe_") as tmp:
        h5_path = Path(tmp) / "choice_action_flow.h5"
        _build_h5(h5_path)
        _assert_record_schema_contract(h5_path)
        print("  schema ✓ choice_action new field point readable, don't write position_3d")

        _assert_resolver_reads_by_is_subgoal_boundary(h5_path)
        print("  resolver ✓ Read only by is_subgoal_boundary + strictly reject legacy fields")

    print("\nPASS: choice_action is_subgoal_boundary flow tests passed")


if __name__ == "__main__":
    main()
