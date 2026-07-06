"""Direct age-standardization of mortality/incidence rates (MSD-III §II.9 measurement).

Turns age-specific counts + person-time into a single comparable rate against a standard
reference population, with a Fay-Feuer gamma confidence interval (the CDC/SEER standard for
directly-standardized rates, well-behaved for the small stratum counts of a rare cancer like
C25). Reference weights are registry-declared (config/registries/health/reference_populations.yaml)
so the standard is auditable, never hard-coded at a call site. Pure numpy/polars; no per-cell loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2

from pegasus.core.config import load_yaml

_REGISTRY = "config/registries/health/reference_populations.yaml"


@dataclass(frozen=True)
class ReferencePopulation:
    name: str
    group_labels: tuple[str, ...]
    lo: np.ndarray                # single-year lower bound per group
    hi: np.ndarray                # inclusive upper bound per group (85+ -> 200)
    weights: np.ndarray           # reference weights per group (any scale; normalized internally)

    def age_group_index(self, age_years: np.ndarray) -> np.ndarray:
        """Vectorized single-year-age -> group index (searchsorted on the lower bounds)."""
        idx = np.searchsorted(self.lo, age_years, side="right") - 1
        return np.clip(idx, 0, len(self.lo) - 1)


def load_reference_population(name: str = "who_world_2000_2025", *, root: str | Path = ".") -> ReferencePopulation:
    reg = load_yaml(Path(root) / _REGISTRY)
    bounds = reg["age_group_bounds"]
    refs = reg["references"]
    if name not in refs:
        raise KeyError(f"unknown reference population '{name}'; have {sorted(refs)}")
    labels = tuple(str(b["group"]) for b in bounds)
    lo = np.array([int(b["lo"]) for b in bounds], dtype=float)
    hi = np.array([int(b["hi"]) for b in bounds], dtype=float)
    w = np.array([float(x) for x in refs[name]["weights"]], dtype=float)
    if len(w) != len(labels):
        raise ValueError(f"reference '{name}' has {len(w)} weights for {len(labels)} age groups")
    return ReferencePopulation(name=name, group_labels=labels, lo=lo, hi=hi, weights=w)


@dataclass(frozen=True)
class StandardizedRate:
    asr: float          # directly age-standardized rate per `per`
    ci_low: float
    ci_high: float
    crude: float        # crude rate per `per`
    deaths: int
    person_years: float
    per: int
    reference: str


def directly_standardized_rate(
    deaths_by_group: np.ndarray,
    person_years_by_group: np.ndarray,
    ref: ReferencePopulation,
    *,
    per: int = 100_000,
    alpha: float = 0.05,
) -> StandardizedRate:
    """Direct-standardized rate + Fay-Feuer (1997) gamma CI.

    deaths/person_years are aligned to ``ref``'s age groups. Groups with zero person-years
    contribute zero (a group absent from the population cannot contribute a rate)."""
    d = np.asarray(deaths_by_group, dtype=float)
    n = np.asarray(person_years_by_group, dtype=float)
    w = ref.weights / ref.weights.sum()
    safe_n = np.where(n > 0, n, np.nan)
    r = np.where(n > 0, d / safe_n, 0.0)                 # age-specific rate
    asr = float(np.nansum(w * r))                         # standardized rate (per 1)
    # Fay-Feuer gamma interval on the standardized rate.
    wn = np.where(n > 0, w / safe_n, 0.0)
    v = float(np.nansum((wn ** 2) * d))                  # variance of ASR
    total_d = int(np.nansum(d))
    if asr <= 0 or v <= 0:
        lo = hi = 0.0
    else:
        lo = (v / (2 * asr)) * chi2.ppf(alpha / 2, df=2 * asr * asr / v)
        w_m = float(np.nanmax(wn))
        hi = ((v + w_m ** 2) / (2 * (asr + w_m))) * chi2.ppf(
            1 - alpha / 2, df=2 * (asr + w_m) ** 2 / (v + w_m ** 2)
        )
    total_n = float(np.nansum(n))
    crude = (total_d / total_n) if total_n > 0 else 0.0
    return StandardizedRate(
        asr=asr * per, ci_low=lo * per, ci_high=hi * per, crude=crude * per,
        deaths=total_d, person_years=total_n, per=per, reference=ref.name,
    )


def standardize_grouped(
    frame: pl.DataFrame,
    *,
    age_col: str,
    deaths_col: str,
    pop_col: str,
    by: list[str] | None = None,
    reference: str = "who_world_2000_2025",
    per: int = 100_000,
    alpha: float = 0.05,
    root: str | Path = ".",
) -> pl.DataFrame:
    """Standardize a long (age x strata) frame of deaths + person-years, one ASR per ``by`` group.

    Vectorized: assigns each single-year age to its reference group, sums deaths/pop per
    (by, group), then applies the Fay-Feuer formula per stratum. Never loops over cells."""
    ref = load_reference_population(reference, root=root)
    by = by or []
    ages = frame[age_col].cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    gidx = ref.age_group_index(np.clip(ages, 0, 200).astype(float))
    binned = frame.with_columns(pl.Series("_grp", gidx))
    agg = (
        binned.group_by(by + ["_grp"])
        .agg(pl.col(deaths_col).sum().alias("_d"), pl.col(pop_col).sum().alias("_n"))
        .sort(by + ["_grp"])
    )
    n_groups = len(ref.weights)
    rows = []
    if by:
        # Scatter the grouped (by, _grp) aggregate into a dense [n_strata, n_groups]
        # deaths/person-years matrix with one numpy fancy-index assignment instead of a
        # Python double loop per stratum. Each (by, _grp) key is unique post group_by, so
        # a direct assignment (not +=) reproduces the previous accumulation exactly.
        strata = agg.select(by).unique().sort(by)
        stratum_index = {tuple(s.values()): i for i, s in enumerate(strata.iter_rows(named=True))}
        n_strata = strata.height
        d_mat = np.zeros((n_strata, n_groups))
        n_mat = np.zeros((n_strata, n_groups))
        key_rows = agg.select(by).iter_rows()
        rid = np.fromiter((stratum_index[k] for k in key_rows), dtype=np.int64, count=agg.height)
        gid = agg["_grp"].cast(pl.Int64).to_numpy()
        d_mat[rid, gid] = agg["_d"].cast(pl.Float64).to_numpy()
        n_mat[rid, gid] = agg["_n"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        for s, i in ((s, stratum_index[tuple(s.values())]) for s in strata.iter_rows(named=True)):
            sr = directly_standardized_rate(d_mat[i], n_mat[i], ref, per=per, alpha=alpha)
            rows.append({**s, "asr": sr.asr, "ci_low": sr.ci_low, "ci_high": sr.ci_high,
                         "crude": sr.crude, "deaths": sr.deaths, "person_years": sr.person_years,
                         "reference": sr.reference, "per": sr.per})
    else:
        d = np.zeros(n_groups); n = np.zeros(n_groups)
        gid = agg["_grp"].cast(pl.Int64).to_numpy()
        d[gid] = agg["_d"].cast(pl.Float64).to_numpy()
        n[gid] = agg["_n"].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
        sr = directly_standardized_rate(d, n, ref, per=per, alpha=alpha)
        rows.append({"asr": sr.asr, "ci_low": sr.ci_low, "ci_high": sr.ci_high,
                     "crude": sr.crude, "deaths": sr.deaths, "person_years": sr.person_years,
                     "reference": sr.reference, "per": sr.per})
    return pl.DataFrame(rows)


__all__ = [
    "ReferencePopulation", "StandardizedRate", "load_reference_population",
    "directly_standardized_rate", "standardize_grouped",
]
