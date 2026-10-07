from __future__ import annotations

"""Shared pytest fixtures."""

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RunConfig, load_config  # noqa: E402


@pytest.fixture(scope="session")
def default_config() -> RunConfig:
    """Canonical config used by every test (config/default.yaml)."""
    return load_config()
