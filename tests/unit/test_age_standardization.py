"""Direct age-standardization (measurement/age_standardization.py, MSD-III §II.9).

Pins the registry-declared reference populations + the Fay-Feuer standardized-rate/CI math.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pegasus.measurement.age_standardization import (
    directly_standardized_rate,
    load_reference_population,
    standardize_grouped,
)


def test_reference_populations_load_and_are_well_formed() -> None:
    for name in ("who_world_2000_2025", "segi_world", "brazil_2010_census"):
        ref = load_reference_population(name)
        assert len(ref.weights) == 18 == len(ref.group_labels)  # 0-4 ... 85+
        assert (ref.weights > 0).all()
    who = load_reference_population("who_world_2000_2025")
    # searchsorted binning: 0->group0, 4->group0, 5->group1, 87->last(85+)
    idx = who.age_group_index(np.array([0.0, 4.0, 5.0, 87.0]))
    assert idx.tolist() == [0, 0, 1, 17]


def test_uniform_age_rate_standardizes_to_the_crude_rate() -> None:
    # If every age group has the SAME rate, the standardized rate equals it regardless of weights.
    ref = load_reference_population("who_world_2000_2025")
    n = np.full(18, 100_000.0)
    d = np.full(18, 50.0)  # 50 per 100k in every group
    sr = directly_standardized_rate(d, n, ref, per=100_000)
    assert abs(sr.asr - 50.0) < 1e-6
    assert abs(sr.crude - 50.0) < 1e-6
    assert sr.ci_low < sr.asr < sr.ci_high


def test_standardization_reweights_an_older_population_down() -> None:
    # A population concentrated in old ages (high crude) standardizes BELOW its crude rate against
    # the younger WHO standard — the whole point of standardization.
    ref = load_reference_population("who_world_2000_2025")
    n = np.array([1.0] * 12 + [100_000.0] * 6)          # people only in the 6 oldest groups
    d = np.array([0.0] * 12 + [500.0] * 6)               # 500/100k in the old groups
    sr = directly_standardized_rate(d, n, ref, per=100_000)
    assert sr.crude > sr.asr                              # crude inflated by the old population
    assert sr.asr > 0


def test_standardize_grouped_matches_the_scalar_path() -> None:
    ref = load_reference_population("who_world_2000_2025")
    df = pl.DataFrame({"age_i": [30, 30, 70, 70], "sex": ["m", "f", "m", "f"],
                       "deaths": [10, 5, 100, 80], "pop": [1_000_000, 1_000_000, 200_000, 250_000]})
    out = standardize_grouped(df, age_col="age_i", deaths_col="deaths", pop_col="pop", by=["sex"])
    assert set(out["sex"].to_list()) == {"m", "f"}
    assert (out["asr"] > 0).all() and (out["ci_low"] <= out["asr"]).all() and (out["asr"] <= out["ci_high"]).all()
