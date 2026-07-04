"""Shared pytest fixtures for the PegaSUS test battery.

Exposes the normalized multi-state DATASUS frames (TDD §5.1) as session-scoped
fixtures so the Phase-0 contract tests can assert against real canonical output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
# Make the fixture helper modules importable by bare name without packaging tests/.
sys.path.insert(0, str(_TESTS_ROOT / "fixtures"))

from datasus_multistate import normalize_all  # noqa: E402


@pytest.fixture(scope="session")
def _normalized_datasus(tmp_path_factory) -> dict:
    """Normalize all four DATASUS systems once per test session."""
    out_dir = tmp_path_factory.mktemp("datasus_normalized")
    return normalize_all(out_dir)


@pytest.fixture
def sim_df(_normalized_datasus):
    return _normalized_datasus["sim"]


@pytest.fixture
def sih_df(_normalized_datasus):
    return _normalized_datasus["sih"]


@pytest.fixture
def sinasc_df(_normalized_datasus):
    return _normalized_datasus["sinasc"]


@pytest.fixture
def cnes_df(_normalized_datasus):
    return _normalized_datasus["cnes"]


@pytest.fixture
def all_df(_normalized_datasus) -> dict:
    return _normalized_datasus


@pytest.fixture
def registry_root() -> str:
    """Path to the on-disk source registry catalog root."""
    return "config/registries"
