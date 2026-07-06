"""Census (9606) + intercensal (6579) population-total stitching (MSD §2.8.10).

Table 9606 (Census demographic matrix) only carries census years; table 6579
(post-censal estimates) carries annual intercensal totals with no disaggregation.
Verified live against IBGE SIDRA (see memory: demographic-tensor-sidra-reorg /
the population-cube intercensal fix): 9606 covers {2010, 2022} only, 6579 covers
2001-2025 annually except 2007 (an IBGE estimation gap), 2010/2022 (census years),
and 2023 (post-census processing lag). These tests use synthetic facts so they
don't depend on network access or IBGE's current publication state.
"""

from pathlib import Path

import polars as pl
import pytest

from pegasus.sidra.facts import normalize_flat_records_to_facts, write_facts_parquet
from pegasus.denominators.population.anchor import (
    load_combined_population_totals_frame,
    load_sidra_intercensal_totals_frame,
)
from pegasus.denominators.population.build import solve_population_tensor_from_sidra_strata

MUNI = "2704302"
MUNI_COD6 = "270430"


def _census_records(year: str, total: float) -> list[dict]:
    return [{
        "table_id": "9606", "variable_id": "93", "period": year, "locality_level": "N6",
        "locality_id": MUNI,
        "classification_tuple": [["2", "Sexo"], ["86", "Cor ou raça"], ["287", "Idade"]],
        "category_tuple": [["2", "6794"], ["86", "95251"], ["287", "100362"]],
        "value": str(total), "unit": "Pessoas",
    }]


def _intercensal_record(year: str, total: float) -> dict:
    return {
        "table_id": "6579", "variable_id": "9324", "period": year, "locality_level": "N6",
        "locality_id": MUNI, "classification_tuple": [], "category_tuple": [],
        "value": str(total), "unit": "Pessoas",
    }


def _write_combined(tmp_path: Path, records: list[dict]) -> Path:
    facts = normalize_flat_records_to_facts(records, table_id="mixed", request_hash="r", metadata_hash="m")
    return write_facts_parquet(facts, output_path=tmp_path / "combined.parquet")


def test_intercensal_loader_reads_6579_totals(tmp_path: Path):
    path = _write_combined(tmp_path, [_intercensal_record("2018", 1000.0), _intercensal_record("2019", 1010.0)])
    frame = load_sidra_intercensal_totals_frame(path)
    assert frame.sort("year").to_dicts() == [
        {"municipality_cod6": MUNI_COD6, "year": 2018, "value": 1000.0},
        {"municipality_cod6": MUNI_COD6, "year": 2019, "value": 1010.0},
    ]


def test_combined_totals_prefers_9606_on_overlap(tmp_path: Path):
    records = [*_census_records("2022", 500.0), _intercensal_record("2021", 480.0), _intercensal_record("2022", 999.0)]
    path = _write_combined(tmp_path, records)
    frame = load_combined_population_totals_frame(path).sort("year")
    assert frame.to_dicts() == [
        {"municipality_cod6": MUNI_COD6, "year": 2021, "value": 480.0},
        {"municipality_cod6": MUNI_COD6, "year": 2022, "value": 500.0},  # 9606 wins, not 6579's 999
    ]


def test_strata_orchestrator_anchors_intercensal_years_from_6579(tmp_path: Path):
    """The core fix: intercensal years (no 9606 disaggregation at all) must still
    get a real closure total from 6579 and a slot on the tensor's period axis --
    not silently vanish, and not fall back to an unconstrained/absent anchor."""
    strata_path = tmp_path / "strata.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(_census_records("2022", 1000.0), table_id="9606", request_hash="r1", metadata_hash="m1"),
        output_path=strata_path,
    )
    totals_records = [
        *_census_records("2022", 1000.0),
        _intercensal_record("2019", 940.0),
        _intercensal_record("2020", 960.0),
        _intercensal_record("2021", 980.0),
    ]
    totals_path = _write_combined(tmp_path, totals_records)

    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path,
        total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet",
        mode="independent_denominator",
    )
    manifest = build.as_manifest()
    # All four years -- the 2022 census year AND the three 6579-only intercensal
    # years -- must appear on the period axis, not just the census year.
    assert set(manifest["periods"]) == {"2019", "2020", "2021", "2022"}

    out = pl.read_parquet(build.output_path)
    for year, expected_total in (("2019", 940.0), ("2020", 960.0), ("2021", 980.0), ("2022", 1000.0)):
        year_rows = out.filter(pl.col("year") == int(year))
        assert year_rows.height > 0, f"missing tensor rows for intercensal year {year}"
        assert year_rows["value"].sum() == pytest.approx(expected_total, abs=0.5)
