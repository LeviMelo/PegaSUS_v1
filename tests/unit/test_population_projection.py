"""FAL-POP-PROJ: native projection beyond official anchors + horizon-growing uncertainty (§II.4).

The projection VALUES come from the solver's cohort-component dynamics (loss.py aging/birth/death)
run on the SV-extrapolated closure past the last census. PROJ adds the prime-directive discipline:
each year is classified by distance from the census anchors, and projected years carry a fragile/
unreliable-typed uncertainty that grows monotonically with that horizon -- so a rate on a projected
denominator is visibly less certain than one on an enumerated or interpolated denominator.
"""

from pathlib import Path

import polars as pl

from pegasus.denominators.population.build import (
    _classify_projection_years,
    solve_population_tensor_from_sidra_strata,
    PROJECTION_H_SOFT,
)
from pegasus.sidra.facts import normalize_flat_records_to_facts, write_facts_parquet

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


def test_classify_projection_years_unit():
    """The classifier: census exact, interpolated bounded, projected accumulates uncertainty."""
    periods = ("1998", "2000", "2010", "2015", "2022", "2023", "2028", "2030")
    census = frozenset({"2000", "2010", "2022"})
    proj, anchored, max_h = _classify_projection_years(periods, census, 0.02)
    assert anchored == (2000, 2022)
    assert proj["2010"][0] == "census" and proj["2010"][3] == 0.02
    assert proj["2015"][0] == "interpolated" and proj["2015"][2] == "verified"
    assert proj["2023"][0] == "projected_forward" and proj["2023"][1] == 1
    assert proj["1998"][0] == "projected_backward" and proj["1998"][1] == 2
    # uncertainty grows monotonically with forward horizon
    assert proj["2023"][3] < proj["2028"][3] < proj["2030"][3]
    # fragile within H_soft, unreliable beyond
    assert proj["2023"][2] == "fragile"  # h=1
    assert proj["2030"][2] == "unreliable"  # h=8 > H_soft=5
    assert max_h == max(2030 - 2022, 2000 - 1998)


def test_no_census_anchor_is_unanchored():
    proj, anchored, max_h = _classify_projection_years(("2021", "2022"), frozenset(), 0.02)
    assert anchored == (None, None) and max_h == 0
    assert all(v[0] == "unanchored" and v[2] == "fragile" for v in proj.values())


def test_tensor_emits_projection_envelope(tmp_path: Path):
    """End-to-end: the tensor parquet carries per-year anchor_class + horizon-growing cell_uncertainty,
    and the build manifest records the anchored range + max horizon."""
    strata = [*_census_records("2010", 1000.0), *_census_records("2022", 1200.0)]
    strata_path = tmp_path / "strata.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(strata, table_id="9606", request_hash="r", metadata_hash="m"),
        output_path=strata_path,
    )
    totals = [
        *_census_records("2010", 1000.0), *_census_records("2022", 1200.0),
        _intercensal_record("2015", 1100.0),   # interpolated
        _intercensal_record("2023", 1210.0),    # projected forward h=1
        _intercensal_record("2024", 1220.0),    # projected forward h=2
    ]
    totals_path = tmp_path / "totals.parquet"
    write_facts_parquet(
        normalize_flat_records_to_facts(totals, table_id="mixed", request_hash="r", metadata_hash="m"),
        output_path=totals_path,
    )
    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path,
        total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet",
        mode="independent_denominator",
    )
    out = pl.read_parquet(build.output_path)
    for col in ("anchor_class", "projection_horizon", "cell_state", "cell_uncertainty"):
        assert col in out.columns, f"missing PROJ column {col}"

    def cls(year):
        return out.filter(pl.col("year") == year)["anchor_class"][0]

    def unc(year):
        return out.filter(pl.col("year") == year)["cell_uncertainty"][0]

    assert cls(2010) == "census" and cls(2022) == "census"
    assert cls(2015) == "interpolated"
    assert cls(2023) == "projected_forward" and cls(2024) == "projected_forward"
    # census years are the most certain; projected uncertainty grows with horizon
    assert unc(2010) < unc(2023) < unc(2024)
    assert out.filter(pl.col("year") == 2023)["cell_state"][0] == "fragile"
    # build manifest surfaces the range for the versioned asset
    assert build.anchored_range == (2010, 2022)
    assert build.max_projection_horizon == 2
    assert set(build.projected_periods) == {"2023", "2024"}
    assert any("population_tensor_projected_years" in w for w in build.result.warnings)
