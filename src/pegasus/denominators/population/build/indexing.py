"""Flat-cell index math for the population tensor (base layer, MSD §2.8).

The (locality, time, age, sex, race) tensor is stored flat/row-major; these helpers
translate a labelled cell to its flat offset, plus a bulk numpy census-count scatter.
The within-group offset ``age*x*r + sex*r + race`` is shared verbatim between the
per-cell ``_cell_index`` and the vectorized ``_census_count_arrays`` -- do not
"centralize" it; the two live here together deliberately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def _cell_index(
    *,
    locality: str,
    period: str,
    age_group: str,
    sex: str,
    race: str,
    locality_index: dict[str, int],
    period_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    shape: tuple[int, int, int, int, int],
) -> int:
    s_count, t_count, a_count, x_count, r_count = shape
    s = locality_index[locality]
    t = period_index[period]
    a = age_index[age_group]
    x = sex_index[sex]
    r = race_index[race]
    return ((((s * t_count) + t) * a_count + a) * x_count + x) * r_count + r


def _birth_cell_index(
    *,
    locality: str,
    period: str,
    sex: str,
    race: str,
    locality_index: dict[str, int],
    period_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    t_count: int,
    x_count: int,
    r_count: int,
) -> int:
    """Index into the birth tensor's ``(locality, time, sex, race)`` flat support.

    Unlike the population tensor itself, births (MSD §2.8.5) have no age axis --
    a newborn always enters at age 0, so ``problem.births`` is shaped one axis
    narrower and indexed independently of ``_cell_index``.
    """
    s = locality_index[locality]
    t = period_index[period]
    x = sex_index[sex]
    r = race_index[race]
    return ((s * t_count + t) * x_count + x) * r_count + r


def _census_count_arrays(
    records_df: pl.DataFrame,
    *,
    locality_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    group_size: int,
    x_count: int,
    r_count: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Per-(locality, census period) dense [group_size] count vector in canonical (age,sex,race) cell
    order. The per-cell offset ``age*x*r + sex*r + race`` matches ``_cell_index``'s within-group layout,
    so a [group_size] vector maps to a contiguous slice of the flat tensor.

    Vectorized (§V.1 / §VIII): labels -> indices via one polars replace_strict per axis, then a single
    numpy scatter (np.add.at) into a dense [n_loc, n_census_period, group_size] buffer (national:
    ~135 MB) -- was a per-record Python loop over the ~12M-row list[dict] (the dominant national
    input-construction cost AND the coexisting-with-n_cells-arrays RAM cliff). Byte-identical: a
    (locality, period) key is emitted iff >=1 valid record touched it (value 0 included, matching the
    old setdefault), and the accumulated sum reproduces the old ``arr[off] += value``."""
    xr = x_count * r_count
    census_periods = sorted({str(p) for p in records_df["period"].unique().to_list()}, key=lambda s: (len(s), s))
    period_to_j = {p: j for j, p in enumerate(census_periods)}
    n_loc, n_cp = len(locality_index), len(census_periods)
    if n_loc == 0 or n_cp == 0:
        return {}
    idx = records_df.select(
        pl.col("municipality_cod6").cast(pl.Utf8).replace_strict(locality_index, default=-1, return_dtype=pl.Int64).alias("li"),
        pl.col("period").cast(pl.Utf8).replace_strict(period_to_j, default=-1, return_dtype=pl.Int64).alias("pj"),
        pl.col("age_group").cast(pl.Utf8).replace_strict(age_index, default=-1, return_dtype=pl.Int64).alias("ai"),
        pl.col("sex").cast(pl.Utf8).replace_strict(sex_index, default=-1, return_dtype=pl.Int64).alias("xi"),
        pl.col("race").cast(pl.Utf8).replace_strict(race_index, default=-1, return_dtype=pl.Int64).alias("ri"),
        pl.col("value").cast(pl.Float64).alias("v"),
    )
    li = idx["li"].to_numpy(); pj = idx["pj"].to_numpy(); ai = idx["ai"].to_numpy()
    xi = idx["xi"].to_numpy(); ri = idx["ri"].to_numpy(); v = idx["v"].to_numpy()
    valid = (li >= 0) & (pj >= 0) & (ai >= 0) & (xi >= 0) & (ri >= 0)
    lp = (li * n_cp + pj)
    flat = lp * group_size + (ai * xr + xi * r_count + ri)
    dense = np.zeros(n_loc * n_cp * group_size, dtype=np.float64)
    np.add.at(dense, flat[valid], v[valid])
    touched = np.zeros(n_loc * n_cp, dtype=bool)
    touched[lp[valid]] = True
    dense = dense.reshape(n_loc, n_cp, group_size)
    out: dict[int, dict[str, np.ndarray]] = {}
    nz_li, nz_j = np.nonzero(touched.reshape(n_loc, n_cp))
    for li_idx, j in zip(nz_li.tolist(), nz_j.tolist()):
        out.setdefault(li_idx, {})[census_periods[j]] = dense[li_idx, j]
    return out


__all__ = [
    "_cell_index",
    "_birth_cell_index",
    "_census_count_arrays",
]
