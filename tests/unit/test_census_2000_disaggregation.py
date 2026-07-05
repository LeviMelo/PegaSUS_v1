"""FAL-POP #3: CTR disaggregation of the 2000 census age brackets → single-year (§II.4).

Pins the demographic-critical invariants: the selected brackets are a clean contiguous partition
of 0..100+, disaggregation preserves each bracket total exactly (closure), and the within-bracket
single-year profile follows the reference (2010) shape — with a safe uniform fallback.
"""

from __future__ import annotations

import numpy as np

from pegasus.sidra.population_cube.census_2000 import (
    CLEAN_AGE_BRACKETS_2093,
    assemble_2000_single_year_records,
    bracket_single_year_labels,
    disaggregate_2000_strata_to_single_year,
    disaggregate_bracket,
)


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
