"""FAL-POP-SV: single-vintage census-anchored intercensal closure (§II.4 anti-discontinuity contract).

SIDRA 6579 carries IBGE's *pre-census projection* vintage; the 2022 census then revised the total
down ~10M nationally. Anchoring 2021 to 6579 and 2022 to the census injects a spurious ~5% denominator
jump that reads as a fake epidemiological trend. With >=2 census anchors the tensor MUST re-anchor
intercensal closure totals to geometric interpolation between the census enumerations (one vintage),
discarding the 6579 value; with <2 anchors it MUST keep the prior 6579 closure (single-anchor fallback).
"""

from pathlib import Path

import polars as pl
import pytest

from pegasus.sidra.facts import normalize_flat_records_to_facts, write_facts_parquet
from pegasus.denominators.population.anchor import geometric_interpolate_closure
from pegasus.denominators.population.build import solve_population_tensor_from_sidra_strata

MUNI = "2704302"


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


def _write(tmp_path: Path, records: list[dict], name: str, table_id: str) -> Path:
    facts = normalize_flat_records_to_facts(records, table_id=table_id, request_hash="r", metadata_hash="m")
    return write_facts_parquet(facts, output_path=tmp_path / name)


def test_geometric_interpolation_removes_national_discontinuity():
    """Unit: the interpolation itself turns the 6579 ~5% drop into a smooth <1% step, exact at census."""
    anchors = {2000: 169_799_170, 2010: 190_755_799, 2022: 203_080_756}  # ~IBGE census totals
    out = geometric_interpolate_closure(anchors, range(2000, 2026))
    # census years exact
    assert out[2000] == anchors[2000] and out[2010] == anchors[2010] and out[2022] == anchors[2022]
    # 2021 lands on the 2010->2022 trajectory (~202M), NOT the 6579 projection (~213M)
    assert 201_000_000 < out[2021] < 203_000_000
    # the operative contract: the 2021->2022 transition is smooth (<1%), not the 6579 ~5% jump
    assert abs(out[2022] - out[2021]) / out[2022] < 0.01
    # monotone rising across the intercensal span, and beyond-census years extrapolate upward
    assert all(out[y] <= out[y + 1] for y in range(2000, 2025))


def test_tensor_reanchors_intercensal_closure_off_6579(tmp_path: Path):
    """End-to-end: with 2010+2022 census anchors, a wildly-off 6579 2021 value is DISCARDED and the
    tensor's 2021 closure comes from census interpolation instead."""
    strata_records = [*_census_records("2010", 1000.0), *_census_records("2022", 1200.0)]
    strata_path = _write(tmp_path, strata_records, "strata.parquet", "9606")
    totals_records = [
        *_census_records("2010", 1000.0),
        *_census_records("2022", 1200.0),
        _intercensal_record("2021", 9999.0),  # deliberately absurd 6579 value -> must be overwritten
    ]
    totals_path = _write(tmp_path, totals_records, "totals.parquet", "mixed")

    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path,
        total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet",
        mode="independent_denominator",
    )
    out = pl.read_parquet(build.output_path)
    total_2021 = out.filter(pl.col("year") == 2021)["value"].sum()
    total_2022 = out.filter(pl.col("year") == 2022)["value"].sum()
    total_2010 = out.filter(pl.col("year") == 2010)["value"].sum()

    expected_2021 = geometric_interpolate_closure({2010: 1000.0, 2022: 1200.0}, [2021])[2021]
    assert total_2021 == pytest.approx(expected_2021, abs=1.0)  # census-interpolated, ~1182
    assert total_2021 < 1500.0  # emphatically NOT the 6579 value of 9999
    assert total_2010 == pytest.approx(1000.0, abs=0.5)  # census year untouched
    assert total_2022 == pytest.approx(1200.0, abs=0.5)  # census year untouched
    # telemetry records the re-anchoring
    assert any("closure_single_vintage_census_anchored" in w for w in build.result.warnings)


def test_single_census_anchor_keeps_prior_closure(tmp_path: Path):
    """Fallback: with <2 census anchors, geometric interpolation is undefined, so the prior (6579)
    closure MUST be preserved rather than fabricated."""
    strata_path = _write(tmp_path, _census_records("2022", 1000.0), "strata.parquet", "9606")
    totals_records = [*_census_records("2022", 1000.0), _intercensal_record("2021", 980.0)]
    totals_path = _write(tmp_path, totals_records, "totals.parquet", "mixed")

    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path,
        total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet",
        mode="independent_denominator",
    )
    out = pl.read_parquet(build.output_path)
    assert out.filter(pl.col("year") == 2021)["value"].sum() == pytest.approx(980.0, abs=0.5)  # 6579 kept
    assert not any("closure_single_vintage_census_anchored" in w for w in build.result.warnings)
