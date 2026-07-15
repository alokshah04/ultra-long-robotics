"""
test_record_stick.py
====================
Verify Stick environment (PatternLock) and non-Stick environment (PickXtimes)
when RecordWrapper records HDF5, whether the following four dimensions are correctly aligned:

1. gripper_state  : Stick → [0.0, 0.0]；non-Stick → shape==(2,)
2. joint_action   : Stick → shape==(8,) and [-1] == -1.0；non-Stick → shape==(8,)
3. eef_action     : Stick → shape==(7,) and [-1] == -1.0；non-Stick → shape==(7,)
4. waypoint_action: shape==(7,)；finite Stick → [-1] == -1.0，non-Stick → ±1.0；
                    non-finite (NaN/Inf) is treated as a "no keypoint" placeholder, skip sign assertion

Test method: refer to generate-dataset-control-seed-readJson-advanceV3.py,
run a complete episode for each test case using FailAware Planner + screw→RRT* retry patch
(with seed retry), then open the generated HDF5 file and assert item by item.

Run (requires display / headless GPU):
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/dataset/test_record_stick.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np
import pytest

from tests._shared.dataset_generation import DatasetCase, DatasetFactoryCache
from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.dataset

# ── Ensure robomme package can be found (compatible with direct main() run) ──────────────────────────────────
_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# ────────────────────────────────────────────────────────────────────────────
# Assertion functions
# ────────────────────────────────────────────────────────────────────────────

def _verify_stick(h5_path: Path, env_id: str):
    """Verify Stick environment HDF5 data assertions."""
    print(f"\n  [Verify Stick] Opening {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        assert len(episode_keys) > 0, "No episode group in HDF5 file"
        ep_grp = f[episode_keys[0]]
        ts_keys = [k for k in ep_grp.keys() if k.startswith("timestep_")]
        assert len(ts_keys) > 0, "No timestep in episode group"

        for ts_key in ts_keys:
            ts = ep_grp[ts_key]

            # 1. gripper_state → [0.0, 0.0]
            gs = np.array(ts["obs"]["gripper_state"])
            assert gs.shape == (2,), \
                f"[{env_id}/{ts_key}] gripper_state shape={gs.shape} expected (2,)"
            assert np.allclose(gs, 0.0), \
                f"[{env_id}/{ts_key}] gripper_state={gs} expected [0.0, 0.0]"

            # 2. joint_action → 8D, last bit == -1.0
            ja = np.array(ts["action"]["joint_action"]).flatten()
            assert ja.shape == (8,), \
                f"[{env_id}/{ts_key}] joint_action shape={ja.shape} expected (8,)"
            assert float(ja[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] joint_action[-1]={ja[-1]} expected -1.0"

            # 3. eef_action → 7D, last bit == -1.0
            ea = np.array(ts["action"]["eef_action"]).flatten()
            assert ea.shape == (7,), \
                f"[{env_id}/{ts_key}] eef_action shape={ea.shape} expected (7,)"
            assert float(ea[-1]) == -1.0, \
                f"[{env_id}/{ts_key}] eef_action[-1]={ea[-1]} expected -1.0"

            # 4. waypoint_action → 7D; non-finite as no keypoint placeholder, finite then verify sign
            wa = np.array(ts["action"]["waypoint_action"]).flatten()
            assert wa.shape == (7,), \
                f"[{env_id}/{ts_key}] waypoint_action shape={wa.shape} expected (7,)"
            if np.all(np.isfinite(wa)):
                assert float(wa[-1]) == -1.0, \
                    f"[{env_id}/{ts_key}] waypoint_action[-1]={wa[-1]} expected -1.0"

    print(f"  [Verify Stick ✓] {env_id} all assertions passed, total {len(ts_keys)} timesteps")


def _verify_non_stick(h5_path: Path, env_id: str):
    """Verify non-Stick environment HDF5 data assertions (original logic not broken)."""
    print(f"\n  [Verify Non-Stick] Opening {h5_path.name}")
    with h5py.File(h5_path, "r") as f:
        episode_keys = [k for k in f.keys() if k.startswith("episode_")]
        assert len(episode_keys) > 0, "No episode group in HDF5 file"
        ep_grp = f[episode_keys[0]]
        ts_keys = [k for k in ep_grp.keys() if k.startswith("timestep_")]
        assert len(ts_keys) > 0, "No timestep in episode group"

        for ts_key in ts_keys:
            ts = ep_grp[ts_key]

            # 1. gripper_state shape == (2,)
            gs = np.array(ts["obs"]["gripper_state"])
            assert gs.shape == (2,), \
                f"[{env_id}/{ts_key}] gripper_state shape={gs.shape} expected (2,)"

            # 2. joint_action → 8D
            ja = np.array(ts["action"]["joint_action"]).flatten()
            assert ja.shape == (8,), \
                f"[{env_id}/{ts_key}] joint_action shape={ja.shape} expected (8,)"

            # 3. eef_action → 7D
            ea = np.array(ts["action"]["eef_action"]).flatten()
            assert ea.shape == (7,), \
                f"[{env_id}/{ts_key}] eef_action shape={ea.shape} expected (7,)"

            # 4. waypoint_action → 7D; non-finite as no keypoint placeholder, finite then verify sign
            wa = np.array(ts["action"]["waypoint_action"]).flatten()
            assert wa.shape == (7,), \
                f"[{env_id}/{ts_key}] waypoint_action shape={wa.shape} expected (7,)"
            if np.all(np.isfinite(wa)):
                assert float(wa[-1]) in (-1.0, 1.0), \
                    f"[{env_id}/{ts_key}] waypoint_action[-1]={wa[-1]} should be ±1.0"

    print(f"  [Verify Non-Stick ✓] {env_id} all assertions passed, total {len(ts_keys)} timesteps")


# ────────────────────────────────────────────────────────────────────────────
# Test case configuration
# ────────────────────────────────────────────────────────────────────────────

# (env_id, is_stick, episode, base_seed, difficulty)
# base_seed is unrelated to the seed corresponding to SOURCE_METADATA_ROOT in the V3 script,
# here directly use the SEED_OFFSET rule of generate_dataset.py
TEST_CASES = [
    ("PatternLock", True,  0, 510001, "easy"),
    ("PickXtimes",  False, 0, 504101, "easy"),
]


def _make_case(env_id: str, episode: int, base_seed: int, difficulty: str | None) -> DatasetCase:
    return DatasetCase(
        env_id=env_id,
        episode=episode,
        base_seed=base_seed,
        difficulty=difficulty,
        save_video=True,
        mode_tag="stick_record_replay",
    )


@pytest.mark.parametrize("env_id,is_stick,episode,base_seed,difficulty", TEST_CASES)
def test_record_stick_case(
    env_id: str,
    is_stick: bool,
    episode: int,
    base_seed: int,
    difficulty: str | None,
    dataset_factory,
):
    generated = dataset_factory(_make_case(env_id, episode, base_seed, difficulty))
    if is_stick:
        _verify_stick(generated.raw_h5_path, env_id)
    else:
        _verify_non_stick(generated.raw_h5_path, env_id)


def main():
    all_pass = True
    results = []

    with tempfile.TemporaryDirectory(prefix="test_record_shared_cache_") as tmpdir:
        cache = DatasetFactoryCache(Path(tmpdir))
        for env_id, is_stick, episode, base_seed, difficulty in TEST_CASES:
            print(f"\n{'='*60}")
            print(f"Test case: {env_id}  (is_stick={is_stick}, ep={episode}, base_seed={base_seed})")
            print(f"{'='*60}")
            try:
                generated = cache.get(_make_case(env_id, episode, base_seed, difficulty))
                if is_stick:
                    _verify_stick(generated.raw_h5_path, env_id)
                else:
                    _verify_non_stick(generated.raw_h5_path, env_id)
                results.append((env_id, "PASS", None))
            except AssertionError as exc:
                results.append((env_id, "FAIL", str(exc)))
                all_pass = False
                print(f"\n  [Assertion failed] {exc}")
                traceback.print_exc()
            except Exception as exc:
                results.append((env_id, "ERROR", str(exc)))
                all_pass = False
                print(f"\n  [Error] {exc}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print("Test results summary")
    print(f"{'='*60}")
    for env_id, status, msg in results:
        marker = "✓" if status == "PASS" else "✗"
        suffix = f"  ({msg})" if msg else ""
        print(f"  {marker} [{status}] {env_id}{suffix}")

    if all_pass:
        print("\n✓ ALL ASSERTIONS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME ASSERTIONS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
