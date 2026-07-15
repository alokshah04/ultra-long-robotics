from __future__ import annotations

from typing import Callable

import pytest

from tests._shared.dataset_generation import (
    DatasetCase,
    DatasetFactoryCache,
    GeneratedDataset,
)


pytestmark = pytest.mark.dataset


@pytest.fixture(scope="session")
def dataset_factory(tmp_path_factory) -> Callable[[DatasetCase], GeneratedDataset]:
    cache_root = tmp_path_factory.mktemp("robomme_dataset_cache", numbered=False)
    cache = DatasetFactoryCache(cache_root)
    return cache.get


@pytest.fixture(scope="session")
def video_unmaskswap_train_ep0_dataset(dataset_factory) -> GeneratedDataset:
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id="VideoUnmaskSwap",
        dataset="train",
        action_space="joint_angle",
        gui_render=False,
    )
    seed, difficulty = builder.resolve_episode(0)
    case = DatasetCase(
        env_id="VideoUnmaskSwap",
        episode=0,
        base_seed=int(seed) if seed is not None else 0,
        difficulty=str(difficulty) if difficulty else None,
        save_video=True,
        mode_tag="obs_train_ep0",
    )
    return dataset_factory(case)

