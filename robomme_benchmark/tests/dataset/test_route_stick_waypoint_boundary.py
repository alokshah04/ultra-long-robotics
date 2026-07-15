from __future__ import annotations

import h5py
import numpy as np
import pytest

from tests._shared.dataset_generation import DatasetCase

pytestmark = pytest.mark.dataset


def _make_case() -> DatasetCase:
    # Keep difficulty fixed to easy to reduce episode variance while validating
    # the demo->non-demo waypoint boundary behavior.
    return DatasetCase(
        env_id="RouteStick",
        episode=0,
        base_seed=16000,
        difficulty="easy",
        save_video=True,
        mode_tag="route_stick_waypoint_boundary",
    )


def _decode_h5_string(raw) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.flatten()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8")
    return str(raw)


def _collect_non_demo_records(ep_group: h5py.Group) -> list[dict]:
    timestep_keys = sorted(
        (k for k in ep_group.keys() if k.startswith("timestep_")),
        key=lambda k: int(k.split("_")[1]),
    )
    out: list[dict] = []
    for key in timestep_keys:
        ts = ep_group[key]
        info = ts["info"]
        is_demo = bool(np.reshape(np.asarray(info["is_video_demo"][()]), -1)[0])
        if is_demo:
            continue
        waypoint_action = np.asarray(ts["action"]["waypoint_action"][()]).flatten()
        subgoal = _decode_h5_string(info["simple_subgoal_online"][()])
        out.append(
            {
                "timestep": int(key.split("_")[1]),
                "subgoal": subgoal,
                "waypoint_action": waypoint_action,
                "is_finite_waypoint": bool(
                    waypoint_action.shape == (7,) and np.all(np.isfinite(waypoint_action))
                ),
            }
        )
    return out


def _unique_finite_waypoints(records: list[dict]) -> list[np.ndarray]:
    uniques: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for item in records:
        if not item["is_finite_waypoint"]:
            continue
        wa = np.asarray(item["waypoint_action"]).flatten()
        if prev is None or not np.array_equal(wa, prev):
            uniques.append(wa.copy())
            prev = wa.copy()
    return uniques


def test_route_stick_first_non_demo_keeps_midpoint(dataset_factory):
    generated = dataset_factory(_make_case())

    with h5py.File(generated.raw_h5_path, "r") as h5f:
        ep_group = h5f["episode_0"]
        non_demo_records = _collect_non_demo_records(ep_group)

    assert non_demo_records, "RouteStick episode_0 has no non-demo records."

    # 1) Boundary regression: first non-demo window should not be all NaN placeholders.
    head_window = non_demo_records[:20]
    assert any(item["is_finite_waypoint"] for item in head_window), (
        "First 20 non-demo timesteps are all NaN waypoints; "
        "demo->non-demo boundary likely dropped the first waypoint."
    )

    # 2) First non-demo subgoal segment should contain at least two unique finite
    # waypoints (midpoint + endpoint), not just endpoint.
    first_subgoal = non_demo_records[0]["subgoal"]
    first_segment: list[dict] = []
    for item in non_demo_records:
        if item["subgoal"] != first_subgoal:
            break
        first_segment.append(item)
    first_segment_unique = _unique_finite_waypoints(first_segment)
    assert len(first_segment_unique) >= 2, (
        "First non-demo subgoal segment has fewer than 2 unique finite waypoints; "
        "expected midpoint + endpoint."
    )

    # 3) Easy RouteStick should have at least 3 unique finite waypoints overall.
    all_unique = _unique_finite_waypoints(non_demo_records)
    assert len(all_unique) >= 3, (
        f"Expected at least 3 unique non-demo waypoints for easy RouteStick, got {len(all_unique)}."
    )
