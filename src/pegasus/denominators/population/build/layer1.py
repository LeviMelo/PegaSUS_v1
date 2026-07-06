"""Layer-1 closed-form prior-mean tensor: census-composition interpolation (MSD §2.8.10)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pegasus.denominators.population.build.indexing import *  # noqa: F401,F403 (intra-package base layer)
from pegasus.denominators.population.build.indexing import _census_count_arrays


def _interpolate_shares(shares: dict[str, np.ndarray], census_years: list[str], t: int) -> np.ndarray:
    """Linear share-space interpolation of the [group_size] composition to year ``t``, clamped to the
    nearest census outside the observed bracket (MSD §2.8.8/§2.8.10 policy)."""
    anchors = [int(y) for y in census_years]
    if t <= anchors[0]:
        return shares[census_years[0]]
    if t >= anchors[-1]:
        return shares[census_years[-1]]
    lo_year = census_years[max(i for i, y in enumerate(anchors) if y <= t)]
    hi_year = census_years[min(i for i, y in enumerate(anchors) if y >= t)]
    if lo_year == hi_year:
        return shares[lo_year]
    w = (t - int(lo_year)) / (int(hi_year) - int(lo_year))
    return (1.0 - w) * shares[lo_year] + w * shares[hi_year]


def interpolate_census_composition(
    *,
    records: list[dict[str, Any]],
    closure: list[float | None],
    locality_index: dict[str, int],
    period_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    shape: tuple[int, int, int, int, int],
) -> np.ndarray:
    """Closed-form prior-mean population tensor (MSD §2.8.10 reconstruction, data-poor limit).

    The intercensal (age,sex,race) breakdown is a *reconstruction*, and in the absence of
    informative flows it is exactly the classical estimate: hold the census joint composition
    and scale it to each year's closure total. This is the deterministic, instant, cacheable
    layer -- ``P^0_{s,t,a,x,r} = E_{s,t} * pi_{s,t,a,x,r}`` where ``pi`` is the locality's
    census (age,sex,race) share linearly interpolated across census years (share-space,
    clamped to the nearest census outside the observed bracket -- same policy as the §2.8.8
    race prior, generalized to the full joint). Census years reproduce their observed strata
    exactly (the shares come from them). Cells with no closure / no census composition fall
    back to 0 (the projection distributes closure over them). This is the solver's warm start
    and, when no flow term is informative, the reconstruction itself.
    """
    s_count, t_count, a_count, x_count, r_count = shape
    group_size = a_count * x_count * r_count
    # Vectorized (§V.1): build one dense [group_size] count vector per (locality, census year) in
    # canonical cell order, then interpolate the WHOLE composition across periods with numpy and write
    # a contiguous slice per (locality, period) -- no per-cell Python loop / _cell_index (was O(n_cells)
    # calls; the dominant national input-construction cost).
    census_counts = _census_count_arrays(
        records, locality_index=locality_index, age_index=age_index, sex_index=sex_index,
        race_index=race_index, group_size=group_size, x_count=x_count, r_count=r_count,
    )
    closure_arr = np.asarray([c if c is not None else np.nan for c in closure], dtype=np.float64)
    values = np.zeros(s_count * t_count * group_size, dtype=np.float64)
    period_pairs = sorted(period_index.items(), key=lambda kv: int(kv[0]))
    for li, by_year in census_counts.items():
        shares = {y: arr / s for y, arr in ((y, arr) for y, arr in by_year.items()) if (s := arr.sum()) > 0}
        census_years = sorted(shares, key=int)
        if not census_years:
            continue
        for period, pi in period_pairs:
            closure_total = closure_arr[li * t_count + pi]
            if not np.isfinite(closure_total) or closure_total <= 0:
                continue
            interp = _interpolate_shares(shares, census_years, int(period))
            base = (li * t_count + pi) * group_size
            values[base:base + group_size] = closure_total * interp
    return values  # numpy array (not .tolist() — a national 1.3e8-element Python list is ~4 GB, §V.1)


__all__ = [
    "_interpolate_shares",
    "interpolate_census_composition",
]
