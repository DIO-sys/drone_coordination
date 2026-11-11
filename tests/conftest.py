from __future__ import annotations

from pathlib import Path

import pytest

from planguard.config import WorldConfig

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg() -> WorldConfig:
    return WorldConfig.load(REPO / "experiments" / "configs" / "world.yaml")


@pytest.fixture(scope="session")
def cases_path() -> Path:
    return REPO / "tests" / "golden" / "cases.yaml"
