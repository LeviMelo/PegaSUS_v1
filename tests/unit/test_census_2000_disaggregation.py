"""FAL-POP #3: CTR disaggregation of the 2000 census age brackets → single-year (§II.4).

Pins the demographic-critical invariants: the selected brackets are a clean contiguous partition
of 0..100+, disaggregation preserves each bracket total exactly (closure), and the within-bracket
single-year profile follows the reference (2010) shape — with a safe uniform fallback.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pegasus.denominators.population.census_2000 import (
    CLEAN_AGE_BRACKETS_2093,
    assemble_2000_single_year_records,
    bracket_single_year_labels,
    carve_pre_census_children,
    carve_pre_census_children_frame,
    disaggregate_2000_strata_to_single_year,
    disaggregate_bracket,
    reconcile_undeclared_race,
)


def _rec(cod6, period, value):
    return {"municipality_cod6": cod6, "period": period, "age_group": "total", "sex": "total", "race": "total", "value": float(value)}


def test_carve_auto_detects_child_via_override_and_preserves_total() -> None:
    """FAL-POP-AMC: a child is auto-detected (present at a later census, absent earlier) and carved out
    of its parents (mass-preserving). Explicit override supplies the parents."""
    # 2000: only the two parents enumerated (child absent). 2010: all three enumerated.
    records = [
        _rec("P1", "2000", 60000.0), _rec("P2", "2000", 40000.0),
        _rec("P1", "2010", 66000.0), _rec("P2", "2010", 44000.0), _rec("C", "2010", 10000.0),
    ]
    total_2000_before = sum(r["value"] for r in records if r["period"] == "2000")
    stats = carve_pre_census_children(records, amc_crosswalk={}, overrides={"C": ["P1", "P2"]})
    assert stats["amc_children_carved"] == 1  # auto-detected, no absent_census_years needed
    assert sum(r["value"] for r in records if r["period"] == "2000") == pytest.approx(total_2000_before)
    child_2000 = sum(r["value"] for r in records if r["municipality_cod6"] == "C" and r["period"] == "2000")
    assert child_2000 == pytest.approx(10000.0 * (100000.0 / 120000.0), rel=1e-9)  # state-ratio back-projection
    assert carve_pre_census_children(records, amc_crosswalk={}, overrides={"C": ["P1", "P2"]})["amc_children_carved"] == 0


def test_carve_resolves_parents_from_amc_group_when_no_override() -> None:
    """The automated path: parents = same-AMC-group members enumerated in the census year (no override).
    A same-group sibling NOT enumerated in the census year is correctly excluded."""
    records = [
        _rec("P1", "2000", 50000.0), _rec("SIB", "2000", 30000.0),       # both in the group, enumerated 2000
        _rec("P1", "2010", 55000.0), _rec("SIB", "2010", 33000.0), _rec("C", "2010", 8000.0),
    ]
    amc = {"P1": 42, "SIB": 42, "C": 42}  # all same AMC group; C installed after 2000
    stats = carve_pre_census_children(records, amc_crosswalk=amc, overrides={})
    assert stats["amc_children_carved"] == 1
    # total preserved; child carved from both group siblings present in 2000
    assert sum(r["value"] for r in records if r["period"] == "2000") == pytest.approx(80000.0)
    assert sum(r["value"] for r in records if r["municipality_cod6"] == "C" and r["period"] == "2000") > 0


def test_columnar_amc_carve_matches_reference_rows() -> None:
    records = [
        _rec("P1", "2000", 60.0), _rec("P2", "2000", 40.0),
        _rec("P1", "2010", 70.0), _rec("P2", "2010", 50.0), _rec("C", "2010", 20.0),
    ]
    reference = [dict(row) for row in records]
    reference_stats = carve_pre_census_children(reference, {}, {"C": ["P1", "P2"]})
    frame, frame_stats = carve_pre_census_children_frame(
        pl.DataFrame(records), {}, {"C": ["P1", "P2"]}
    )
    assert frame_stats == reference_stats
    assert frame.rows() == pl.DataFrame(reference).select(frame.columns).rows()


def test_reconcile_undeclared_race_preserves_total_by_local_composition() -> None:
    """FAL-POP-RECON (§II.5): undeclared-race mass is reallocated into the declared races by the
    local (locality, sex, bracket) composition — total preserved, nothing dropped."""
    # locality M1, sex M, bracket "1140": declared branca=60, parda=40 (60/40 split); undeclared=10.
    declared = {
        ("M1", "male", "branca"): {"1140": 60.0},
        ("M1", "male", "parda"): {"1140": 40.0},
    }
    undeclared = {("M1", "male"): {"1140": 10.0}}
    stats = reconcile_undeclared_race(declared, undeclared)
    assert stats["undeclared_reallocated"] == 10.0
    # 10 undeclared split 60/40 -> branca 66, parda 44; total 110 (was 100 declared + 10 undeclared)
    assert declared[("M1", "male", "branca")]["1140"] == 66.0
    assert declared[("M1", "male", "parda")]["1140"] == 44.0
    total = sum(v for prof in declared.values() for v in prof.values())
    assert total == 110.0  # complete enumerated count; the 10 undeclared were reallocated, not dropped


def test_reconcile_undeclared_race_hierarchical_fallback() -> None:
    """A cell that is entirely undeclared (no local declared mass) borrows a broader composition,
    never dropping the mass."""
    declared = {
        ("M1", "male", "branca"): {"1141": 80.0},   # composition known at (loc,sex) but not bracket 1140
        ("M1", "male", "parda"): {"1141": 20.0},
    }
    undeclared = {("M1", "male"): {"1140": 50.0}}    # bracket 1140 has zero declared -> fallback
    stats = reconcile_undeclared_race(declared, undeclared)
    assert stats["undeclared_reallocated"] == 50.0
    assert stats["fallback_cells"] == 1
    # falls back to the (loc,sex) composition 80/20
    assert declared[("M1", "male", "branca")]["1140"] == 40.0
    assert declared[("M1", "male", "parda")]["1140"] == 10.0
    total = sum(v for prof in declared.values() for v in prof.values())
    assert total == 150.0  # 100 declared + 50 reallocated


def test_clean_partition_is_contiguous_and_covers_0_to_100plus() -> None:
    spans = sorted(CLEAN_AGE_BRACKETS_2093.values())
    # contiguous: each bracket starts exactly where the previous ended + 1; no gaps, no overlaps.
    assert spans[0][0] == 0
    for (lo0, hi0), (lo1, _hi1) in zip(spans, spans[1:]):
        assert lo1 == hi0 + 1, f"gap/overlap between {hi0} and {lo1}"
    assert spans[-1][1] >= 100  # open-ended top bracket reaches age_100_plus

    # union of single-year labels == the full canonical axis age_0..age_99 + age_100_plus, no dupes.
    labels: list[str] = []
    for lo, hi in CLEAN_AGE_BRACKETS_2093.values():
        labels += bracket_single_year_labels(lo, hi)
    expected = [f"age_{a}" for a in range(100)] + ["age_100_plus"]
    assert sorted(set(labels)) == sorted(set(expected))
    assert len(labels) == len(set(labels)) == 101  # every single year covered exactly once


def test_disaggregation_preserves_bracket_total_and_follows_shape() -> None:
    # 30-39 bracket total 1000, reference 2010 shape concentrated in the early-30s.
    ref = {f"age_{a}": float(v) for a, v in zip(range(30, 40), [300, 200, 150, 100, 80, 60, 40, 30, 25, 15])}
    dis = disaggregate_bracket(1000.0, 30, 39, ref)
    assert not dis.used_uniform_fallback
    assert sum(dis.by_age.values()) == 1000.0  # closure exact
    # proportional to the reference: age_30 gets 300/1000 of the shape mass → 300 people.
    assert dis.by_age["age_30"] == 300.0
    assert dis.by_age["age_39"] == 15.0
    # monotone where the shape is monotone
    assert dis.by_age["age_30"] > dis.by_age["age_31"] > dis.by_age["age_39"]


def test_open_top_bracket_spans_to_100_plus() -> None:
    ref = {f"age_{a}": 1.0 for a in range(80, 100)} | {"age_100_plus": 1.0}
    dis = disaggregate_bracket(210.0, 80, 200, ref)
    assert "age_100_plus" in dis.by_age
    assert min(dis.by_age.keys(), key=lambda s: 0) is not None
    assert sum(dis.by_age.values()) == 210.0  # closure across 80..99 + 100_plus (21 labels)
    assert len(dis.by_age) == 21


def test_uniform_fallback_when_reference_empty() -> None:
    dis = disaggregate_bracket(100.0, 20, 24, reference_shape={})  # no 2010 mass
    assert dis.used_uniform_fallback
    assert sum(dis.by_age.values()) == 100.0
    assert all(abs(v - 20.0) < 1e-9 for v in dis.by_age.values())  # 5 single years, uniform


def test_full_profile_disaggregation_ignores_rollups_and_preserves_total() -> None:
    # a (locality,sex,race) 2000 profile over the clean partition + a roll-up that must be ignored.
    bracket_counts = {"1140": 500.0, "1141": 400.0, "100402": 999999.0}  # 100402 = 0-14 roll-up
    ref = {f"age_{a}": 1.0 for a in range(15)}
    out = disaggregate_2000_strata_to_single_year(bracket_counts=bracket_counts, reference_single_year=ref)
    assert abs(sum(out.values()) - 900.0) < 1e-9  # only the two clean brackets, roll-up excluded
    assert set(out.keys()) == {f"age_{a}" for a in range(10)}  # 0-4 + 5-9


def test_assemble_records_preserves_totals_and_uses_locality_fallback() -> None:
    # muni "A", female/parda has a specific 2010 shape; female/branca has NO specific 2010 → falls
    # back to A's locality-marginal 2010 shape (still resolves, closure preserved).
    profiles = {
        ("A", "female", "parda"): {"1140": 100.0},   # 0-4
        ("A", "female", "branca"): {"1140": 50.0},
    }
    reference_2010 = {
        ("A", "female", "parda"): {f"age_{a}": float(v) for a, v in zip(range(5), [5, 4, 3, 2, 1])},
    }
    recs = assemble_2000_single_year_records(bracket_profiles=profiles, reference_2010=reference_2010)
    by_group: dict[tuple[str, str], float] = {}
    for r in recs:
        assert r["period"] == "2000" and r["age_group"].startswith("age_")
        by_group[(r["sex"], r["race"])] = by_group.get((r["sex"], r["race"]), 0.0) + r["value"]
    assert abs(by_group[("female", "parda")] - 100.0) < 1e-9   # closure per cell
    assert abs(by_group[("female", "branca")] - 50.0) < 1e-9   # fallback shape still preserves total
