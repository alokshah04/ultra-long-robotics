from __future__ import annotations

from pathlib import Path

import pytest

from tests._shared.repo_paths import ensure_src_on_path, find_repo_root

REPO_ROOT = ensure_src_on_path(__file__)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_root(repo_root: Path) -> Path:
    return repo_root / "src"


def pytest_configure(config) -> None:
    # Fallback marker registration even if pytest is invoked without pyproject parsing.
    config.addinivalue_line("markers", "slow: slow-running tests")
    config.addinivalue_line("markers", "gpu: tests requiring GPU/display/headless rendering stack")
    config.addinivalue_line("markers", "dataset: tests that generate/use temporary datasets")
    config.addinivalue_line("markers", "lightweight: tests that do not require generated dataset")

