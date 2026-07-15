"""
test_replay_stick.py
====================
Verify Stick environment (PatternLock) and non-Stick environment (PickXtimes)
when being parsed and replayed by EpisodeDatasetResolver (used in dataset_replay.py),
whether various dimensions and states are aligned as expected.
Similar to test_record_stick.py, we will first run one or two complete episodes to ensure a correct HDF5 file is available locally.
Then use BenchmarkEnvBuilder combined with EpisodeDatasetResolver for replay read and assertion against obs.

1. gripper_state(eef_state_list and obs read): Stick -> [0.0, 0.0]; non-Stick -> shape(2,)
2. action (eef / joint_action read from resolver): end-effector dimensions aligned
3. etc.

Run (requires display / headless GPU):
    cd /data/hongzefu/robomme_benchmark
    uv run python tests/dataset/test_replay_stick.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pytest

from tests._shared.dataset_generation import DatasetCase, DatasetFactoryCache
from tests._shared.repo_paths import find_repo_root

pytestmark = pytest.mark.dataset

# ── Ensure robomme package can be found ──────────────────────────────────────────────────
_PROJECT_ROOT = find_repo_root(__file__)
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from robomme.env_record_wrapper import BenchmarkEnvBuilder, EpisodeDatasetResolver  # noqa: E402
from robomme.robomme_env import *  # noqa: F401,F403,E402  Register all custom environments


# ────────────────────────────────────────────────────────────────────────────
# Assertion functions (replay test phase)
# ────────────────────────────────────────────────────────────────────────────
def _verify_replay(env_id: str, dataset_dir: Path, h5_path: Path, is_stick: bool):
    """Verify whether the states read from Builder and Resolver comply with expected model dimension rules"""

    # We know we just recorded ep 0
    replay_episode = 0
    # Note that in Dataset Resolver, we scan the folder where the H5 file is located, because retrying may cause the suffix Seed to change.
    # Parsing can still automatically find the first HDF5 file corresponding to the dataset_dir parameter.

    ACTION_SPACE = "joint_angle"
    print(f"\n  [Start Replay Validation] ACTION_SPACE: {ACTION_SPACE}, env: {env_id}")
    env_builder = BenchmarkEnvBuilder(
        env_id=env_id,
        dataset="test",  # Does not actually use dataset json scan, just acts as a placeholder
        action_space=ACTION_SPACE,
        gui_render=False,
    )

    # Bypass resolver json's episode reduction, directly search local files via dataset resolver.
    env = env_builder.make_env_for_episode(
        replay_episode,
        max_steps=1000,
        include_maniskill_obs=True,
        include_front_depth=True,
        include_wrist_depth=True,
        include_front_camera_extrinsic=True,
        include_wrist_camera_extrinsic=True,
        include_available_multi_choices=True,
        include_front_camera_intrinsic=True,
        include_wrist_camera_intrinsic=True,
    )

    # The unified data generation fixture is pre-prepared with the record_dataset_{env_id}.h5 naming.

    # Create the parser (needs to point to the directory above dataset_dir/hdf5_files, which is dataset_dir)
    try:
        dataset_resolver = EpisodeDatasetResolver(
            env_id=env_id,
            episode=replay_episode,
            dataset_directory=str(dataset_dir),
        )
    except Exception as e:
        env.close()
        raise e

    try:
        obs, info = env.reset()

        step_id = 0
        while True:
            # ======= Get and verify action in Dataset =======
            action = dataset_resolver.get_step(ACTION_SPACE, step_id)
            if action is None:
                break

            eef_action = dataset_resolver.get_step("ee_pose", step_id)
            waypoint_action = dataset_resolver.get_step("waypoint", step_id)
            joint_action = dataset_resolver.get_step("joint_angle", step_id)

            if is_stick:
                assert float(joint_action[-1]) == -1.0, f"[{env_id}] joint_action[-1]={joint_action[-1]} expected -1.0"
                if eef_action is not None:
                     assert float(eef_action[-1]) == -1.0, f"[{env_id}] eef_action[-1]={eef_action[-1]} expected -1.0"
                if waypoint_action is not None and len(waypoint_action) > 0:
                     waypoint_action = np.asarray(waypoint_action).flatten()
                     assert waypoint_action.shape == (7,), f"[{env_id}] waypoint_action shape expected (7,)"
                     assert np.all(np.isfinite(waypoint_action)), f"[{env_id}] waypoint_action should be finite"
                     assert float(waypoint_action[-1]) == -1.0, f"[{env_id}] waypoint_action[-1]={waypoint_action[-1]} expected -1.0"
            else:
                if waypoint_action is not None and len(waypoint_action) >0:
                    waypoint_action = np.asarray(waypoint_action).flatten()
                    assert waypoint_action.shape == (7,), f"[{env_id}] waypoint_action shape expected (7,)"
                    assert np.all(np.isfinite(waypoint_action)), f"[{env_id}] waypoint_action should be finite"
                    assert float(waypoint_action[-1]) in (-1.0, 1.0), f"[{env_id}] expected ±1 for waypoint_action"

            # ======= Execute step =======
            obs, reward, terminated, truncated, info = env.step(action)

            # ======= Assert the obs state returned by DemonstrationWrapper =======
            gripper_state_list = obs["gripper_state_list"]

            if is_stick:
                for gs in gripper_state_list:
                    assert gs.shape == (2,), f"[{env_id}] gripper_state shape expected (2,)"
                    assert np.allclose(gs, 0.0), f"[{env_id}] gripper_state expected [0.0, 0.0] but got {gs}"
            else:
                for gs in gripper_state_list:
                    assert gs.shape == (2,), f"[{env_id}] gripper_state shape expected (2,)"

            step_id += 1
            if truncated.item() or terminated.item():
                break

    finally:
        env.close()

    print(f"  [{env_id} - Replay Validation ✓] All assertions passed, totally replayed {step_id} timesteps")


# ────────────────────────────────────────────────────────────────────────────
# Test flow control
# ────────────────────────────────────────────────────────────────────────────
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
def test_replay_stick_case(
    env_id: str,
    is_stick: bool,
    episode: int,
    base_seed: int,
    difficulty: str | None,
    dataset_factory,
):
    generated = dataset_factory(_make_case(env_id, episode, base_seed, difficulty))
    _verify_replay(
        env_id=env_id,
        dataset_dir=generated.resolver_dataset_dir,
        h5_path=generated.raw_h5_path,
        is_stick=is_stick,
    )


def main():
    all_pass = True
    results = []

    with tempfile.TemporaryDirectory(prefix="test_replay_shared_cache_") as tmpdir:
        cache = DatasetFactoryCache(Path(tmpdir))
        for env_id, is_stick, episode, base_seed, difficulty in TEST_CASES:
            print(f"\n{'='*60}")
            print(f"Test case: {env_id}  (is_stick={is_stick}, ep={episode})")
            print(f"{'='*60}")
            try:
                generated = cache.get(_make_case(env_id, episode, base_seed, difficulty))
                print(f"  => Recording complete: {generated.raw_h5_path}")
                print(f"  [2. Replay Parsing Phase]")
                _verify_replay(
                    env_id=env_id,
                    dataset_dir=generated.resolver_dataset_dir,
                    h5_path=generated.raw_h5_path,
                    is_stick=is_stick,
                )
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
