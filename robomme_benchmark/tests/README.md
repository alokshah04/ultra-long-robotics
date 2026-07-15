# RoboMME Benchmark Test Instructions

These tests cover logical assertions, action replay, and dataset recording correctness under this benchmark framework. They are mainly divided into two major sections: **`tests/dataset`** (focusing on low-level environment and dataset interaction) and **`tests/lightweight`** (focusing on lightweight logic branches and unit tests).

The following explains the functions implemented by each test and how to run them.

## 1. `dataset/` Directory: Environment Interaction and Dataset Alignment Tests

Tests in this directory mainly verify low-level environment calls based on the physics engine, Wrapper observation packaging for reinforcement learning, and dimensional alignment for massive dataset recording and replay.

*   **`test_obs_config.py`**: Verifies the `include_*` control switches passed in `make_env_for_episode` (e.g., turning on/off front depth map, wrist camera intrinsics/extrinsics, etc.). Tests whether it can return the corresponding fields in `obs` and `info` on demand during `reset()` and `step()` without error.
*   **`test_obs_numpy.py`**: Verifies the correctness of data conversion handled by `DemonstrationWrapper`. This test checks that in the generated temporary dataset, the values in the native `obs` and `info` dictionaries are correctly converted to compliant NumPy ndarray data types and have the expected data structure shapes.
*   **`test_record_stick.py`**: Verifies that when recording demonstrations as HDF5 dataset format using `RecordWrapper`, tasks with special trajectory requirements (like `PatternLock`/`RouteStick`) and normal requirements (like `PickXtimes`) store their gripper state (`gripper_state`), robot arm joints (`joint_action`), and end-effector pose (`eef_action`) correctly with proper dimensions.
*   **`test_replay_stick.py`**: Reverse verification test. Used to read tests: verify whether the special or normal datasets generated in the previous step can align as precisely as when recorded when parsed and replayed by `EpisodeDatasetResolver`.
*   **`test_eepose_error_handling.py`**: Heavy environment interaction test. Verifies that when an end-effector pose target (`ee_pose` action space) beyond the robot arm's reach is passed in, `DemonstrationWrapper` can gracefully catch the underlying physics engine or IK solver errors, and report the exception information by returning `info["status"] = "error"` to avoid simulation program crashes.
*   **`test_route_stick_waypoint_boundary.py`**: Specific route task verification. Ensures that the first online waypoint recorded has sufficient fidelity when the generated demonstration data transitions from offline demonstration to online interaction (Demo -> Non-demo) boundary.
*   **`test_waypoint_phase_isolation.py`**: Verifies data isolation for action commands (especially Waypoints) between demonstration recording and online interaction, preventing residual demonstration actions in the buffer from polluting the recorded data during the online phase.

### Dataset Generation Sharing Mechanism (Pytest Fixture + Cache)

Because rendering and calling the underlying motion planning solver to record qualified datasets is extremely time-consuming, the tests under `tests/dataset/` use a comprehensive data generation caching mechanism (`_shared/dataset_generation.py`) to ensure that a complete demonstration trajectory for the same case is only generated once:

1.  **Session-level generator (`dataset_factory`)**: In `dataset/conftest.py`, a globally unique factory function `dataset_factory` with a lifecycle spanning the entire Test Session is defined.
2.  **Hash-based temporary directory cache (`DatasetFactoryCache`)**: The dataset will be flushed to the temporary directory mechanism `tmp_path_factory` provided by Pytest upon the first request. The unique features of this data are assembled into a `cache_key` by environment name, number of steps, difficulty, and even action control mode.
3.  **Direct retrieval by subsequent tests**: For example, if multiple tests are asking for the `video_unmaskswap_train_ep0_dataset` fixture which contains pre-recorded data, the engine will detect the hit and directly return the same prepared HDF5 test file. This avoids repeating physics calculations for similar task trajectories across multiple test cases.

## 2. `lightweight/` Directory: Lightweight Functional Unit Tests

Mainly targets unit-level or branch-level assertion verification for some internal specific logic such as label matching, data post-processing, and state in various specific task scenarios.

*   **`test_ChoiceLabel.py`**: Tests replay matching logic during action inference (`oracle_action_matcher`), verifying processes such as "accurate extraction of option labels", "ignoring empty label text", and correct mapping to target dictionary options.
*   **`test_ChoicePositionNearest.py`**: Position matching logic test. Covers the 3D nearest neighbor behavior of `select_target_with_position`, including skipping invalid candidates, returning `None` for no valid input, and stably selecting according to flattened candidate order when equidistant.
*   **`test_choice_action_pixel_mapping.py`**: Pixel-level mapping and selection logic. Tests the mapper algorithm (`project_world_to_pixel`) from world 3D coordinates projected to the camera pixel 2D plane, and verifies the accurate nearest selection capability of selecting targets at screen pixel level (`select_target_with_pixel`).
*   **`test_StopcubeIncrement.py`**: Timing function verification for the specific task `StopCube`. Verifies whether the absolute time step (`absTimestep`) increment of the internal scheduler increments as expected and eventually reaches the upper limit phase (Saturation) when the "remain static" option is triggered. Includes cases where simulating backward time steps can correctly reset the counter.
*   **`test_TaskGoal.py` / `test_TaskGoalI_isList.py`**: Branch coverage test for the internal natural language description generation (`get_language_goal`) logic of `task_goal.py`. Verifies that up to a dozen subtasks can assemble accurate quantities of bilingual goal descriptions for specific scenarios.
*   **`test_choice_action_is_keyframe_flow.py`**: Workflow test targeted at features extracted from discrete items and pixel positions. Determines whether its recording truly satisfies the set keyframe admission conditions, and ensures `position_3d` is only recorded as a supplementary field.
*   **`test_waypoint_dense_dedup.py`**: Tests dense trajectory filtering and adjacent dedup matching logic based on the demonstrated waypoint (Waypoint) action space.
*   **`test_record_info_is_completed.py`**: Lightweight validation. Parses the AST tree to ensure `RecordWrapper` correctly handles progress tasks in the online phase (e.g., progress marker field `is_completed`, etc.) during HDF5 file generation.
*   **`test_record_video_metadata_fields.py`**: Lightweight metadata fields check. Scans syntax tree to ensure `RecordWrapper` writes correct command labels, action options, completion flags, etc., to the data record Buffer for replay rendering and visualization verification calls.
*   **`test_record_waypoint_pending_flow.py`**: Syntax flow analysis. Ensures that data flow (Waypoint Pending state updates, caching) during the recording process has correct lifecycle management logic and isolation clearance measures.
*   **`test_step_error_handling.py`**: Lightweight structural test and syntax analysis. Uses Mock and AST checks to confirm the core environment wrapper layer can effectively catch errors, set `info["status"]` to `"error"`; and verifies if other callers (like `run_example.py`) correctly caught the safety error flag, replacing rigid `try-except` blocks.

## 3. Public Settings and Helper Scripts (`conftest.py` and `_shared/`)

*   **`conftest.py` & `dataset/conftest.py`**: Defines Pytest Fixtures at all levels (including how to pre-register related environments via `BenchmarkEnvBuilder`, and build dedicated temporary storage factories).
*   **`_shared/`**: Contains utility scripts like `dataset_generation.py` used to coordinate with temporary HDF5 mock structures and uniformly manage the benchmark project path locations.

## How to Run Tests

This project heavily relies on **`uv`** to manage virtual environments. All test execution commands must be guided by `uv run` from the code root directory.

### 1. Run all tests

```bash
uv run python -m pytest tests/
```

If you want to see real-time `print()` and environment building standard output prompts, you can turn off log capture via `-s`:

```bash
uv run python -m pytest tests/ -s
```

### 2. Run by section

**Run non-physics rendering tests leaning towards pure logical verification (extremely fast):**

```bash
uv run python -m pytest tests/lightweight/
```

**Run tests that need actual Mujoco simulation physics and data loading wrappers (slightly time-consuming):**

```bash
uv run python -m pytest tests/dataset/
```

### 3. Execute a specific test script or single test method

Down to the file:

```bash
uv run python -m pytest tests/lightweight/test_TaskGoal.py
```

Run a single test case under a file (for example):

```bash
uv run python -m pytest tests/lightweight/test_TaskGoal.py::test_binfill_two_colors
```

### 4. Run via Decorator Marks (Pytest Mark)

For some files given `@pytest.mark.dataset`, you can also execute via match:

```bash
uv run python -m pytest -m dataset
```
