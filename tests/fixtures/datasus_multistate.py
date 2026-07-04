"""Multi-state DATASUS fixtures for the Phase-0 contract battery (TDD §5.1, T0-1).

The four raw CSV fixtures under ``tests/fixtures/datasus/`` are hand-authored so that
each system carries records exercising a spread of MSD-I §2.3 field states
(``valid`` / ``missing`` / ``invalid`` / ``unparseable`` / ``unknown`` / sentinel).
This module is the single place that turns those raw CSVs into the canonical,
normalized SHE frames the contract tests assert against — via the *production*
vectorized normalizers (``pegasus.datasus.normalize``), never a bespoke decode path.

Import surface:
    RAW_FIXTURES            -> {system: Path to raw CSV}
    normalize_system(sys, out_dir) -> pl.DataFrame  (one normalized frame)
    normalize_all(out_dir)  -> {system: pl.DataFrame}
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.datasus.normalize import (
    normalize_cnes_st_events,
    normalize_sih_rd_events,
    normalize_sim_do_events,
    normalize_sinasc_events,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "datasus"

RAW_FIXTURES: dict[str, Path] = {
    "sim": FIXTURE_DIR / "sim_do_fixture.csv",
    "sih": FIXTURE_DIR / "sih_rd_fixture.csv",
    "sinasc": FIXTURE_DIR / "sinasc_fixture.csv",
    "cnes": FIXTURE_DIR / "cnes_st_fixture.csv",
}

_NORMALIZERS = {
    "sim": normalize_sim_do_events,
    "sih": normalize_sih_rd_events,
    "sinasc": normalize_sinasc_events,
    "cnes": normalize_cnes_st_events,
}

_MANIFEST_HASH = "multistate_fixture_manifest"


def normalize_system(system: str, out_dir: str | Path) -> pl.DataFrame:
    """Normalize one system's raw fixture into its canonical SHE frame."""
    if system not in RAW_FIXTURES:
        raise KeyError(f"unknown system {system!r}; known: {sorted(RAW_FIXTURES)}")
    out_path = Path(out_dir) / f"{system}_normalized.parquet"
    _NORMALIZERS[system](
        input_path=str(RAW_FIXTURES[system]),
        output_path=str(out_path),
        source_manifest_hash=_MANIFEST_HASH,
    )
    return pl.read_parquet(out_path)


def normalize_all(out_dir: str | Path) -> dict[str, pl.DataFrame]:
    """Normalize every system; returns {system: frame}."""
    return {system: normalize_system(system, out_dir) for system in RAW_FIXTURES}
