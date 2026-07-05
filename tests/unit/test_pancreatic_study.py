"""Pancreatic cancer (C25) study engine — core extraction + assembly (studies/)."""

from __future__ import annotations

import polars as pl

from pegasus.studies.pancreatic_cancer import build_pancreatic_study, extract_pancreatic_deaths


def _sim_fixture() -> pl.DataFrame:
    return pl.DataFrame({
        "underlying_icd_norm": ["C259", "C250", "I219", "C25", "J189", "C61"],  # 3 C25, 3 not
        "year": [2020, 2020, 2020, 2021, 2021, 2021],
        "sex": ["male", "female", "male", "male", "female", "male"],
        "age_years": [72.0, 65.0, 55.0, 80.0, 44.0, 70.0],
        "race_color_admin": ["4", "1", "2", "4", "1", "1"],
        "mun_residence_cod6": ["270430", "270430", "355030", "355030", "270630", "270630"],
    })


def test_extract_only_c25() -> None:
    d = extract_pancreatic_deaths(_sim_fixture())
    assert d.height == 3                                    # C259, C250, C25 — not I21/J18/C61
    assert set(d["underlying_icd_norm"].to_list()) == {"C259", "C250", "C25"}


def test_study_assembles_real_structure() -> None:
    study = build_pancreatic_study(_sim_fixture(), total_population=1_000_000)
    assert study.n_deaths == 3
    assert study.crude_rate_per_100k == 0.3                 # 3 / 1e6 * 1e5
    assert {r["group"] for r in study.by_sex} == {"male", "female"}
    # C25 spans two UFs (27 Alagoas / 35 São Paulo) → Nordeste + Sudeste regions
    assert {r["group"] for r in study.by_region} == {"Nordeste", "Sudeste"}
    # the disease axis classifies C25 into its neoplasm chapter/block
    assert "chapter_II" in study.disease_concepts and "block_C15-C26" in study.disease_concepts
