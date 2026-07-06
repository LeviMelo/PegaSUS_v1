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
    records: list[dict[str, Any]],
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
    order. One O(records) pass; the per-cell offset ``age*x*r + sex*r + race`` matches ``_cell_index``'s
    within-group layout, so a [group_size] vector maps to a contiguous slice of the flat tensor."""
    xr = x_count * r_count
    out: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        li = locality_index.get(record["municipality_cod6"])
        ai = age_index.get(record["age_group"])
        xi = sex_index.get(record["sex"])
        ri = race_index.get(record["race"])
        if li is None or ai is None or xi is None or ri is None:
            continue
        by_year = out.setdefault(li, {})
        arr = by_year.get(record["period"])
        if arr is None:
            arr = np.zeros(group_size, dtype=np.float64)
            by_year[record["period"]] = arr
        arr[ai * xr + xi * r_count + ri] += float(record["value"])
    return out


__all__ = [
    "_cell_index",
    "_birth_cell_index",
    "_census_count_arrays",
]
