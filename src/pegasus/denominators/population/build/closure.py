"""Closure panel: vital totals, net-migration residual, and single-vintage re-anchoring (MSD §2.8)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.denominators.population.anchor import geometric_interpolate_closure

from pegasus.denominators.population.build.priors import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.priors import _resolve_geo_year_columns


def _sidra_vital_totals(path: str | Path | None, *, table_id: str, variable_id: str) -> dict[tuple[str, str], float] | None:
    """Per-(municipality_cod6, year) total from a SIDRA civil-registry facts file.

    The request that produced ``path`` is total-only (every classification pinned
    to its Total category), so each locality-year is one numeric row; summed for
    safety. Returns ``None`` when the file is absent so the caller can fall back to
    a DATASUS event source.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    frame = pl.read_parquet(p)
    required = {"table_id", "variable_id", "period", "locality_id", "value_numeric", "value_status"}
    if required - set(frame.columns):
        return None
    frame = frame.filter(
        (pl.col("table_id").cast(pl.Utf8) == table_id)
        & (pl.col("variable_id").cast(pl.Utf8) == variable_id)
        & (pl.col("value_status").cast(pl.Utf8) == "numeric")
        & pl.col("value_numeric").is_not_null()
    )
    out: dict[tuple[str, str], float] = {}
    for row in frame.group_by([pl.col("locality_id").cast(pl.Utf8).str.slice(0, 6), pl.col("period").cast(pl.Utf8).str.slice(0, 4)]).agg(
        pl.col("value_numeric").sum().alias("total")
    ).iter_rows(named=True):
        out[(str(row["locality_id"]), str(row["period"]))] = float(row["total"])
    return out


def _datasus_event_totals(
    path: str | Path | None, *, geo_candidates: tuple[str, ...], year_candidates: tuple[str, ...]
) -> dict[tuple[str, str], float] | None:
    """Per-(municipality_cod6, year) event COUNT from a DATASUS event file (fallback
    vital source for the migration residual when no civil-registry facts exist)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    frame = _resolve_geo_year_columns(pl.read_parquet(p), geo_candidates=geo_candidates, year_candidates=year_candidates)
    if frame is None:
        return None
    out: dict[tuple[str, str], float] = {}
    for row in frame.group_by(["municipality_cod6", "year"]).agg(pl.len().cast(pl.Float64).alias("total")).iter_rows(named=True):
        if row["municipality_cod6"] is None or row["year"] is None:
            continue
        out[(str(row["municipality_cod6"]), str(int(row["year"])))] = float(row["total"])
    return out


def _migration_residual_totals(
    *,
    closure: list[float | None],
    births_by_st: dict[tuple[str, str], float] | None,
    deaths_by_st: dict[tuple[str, str], float] | None,
    localities: tuple[str, ...],
    periods: tuple[str, ...],
    shape: tuple[int, int, int, int, int],
) -> tuple[tuple[float | None, ...] | None, list[str]]:
    """Net-migration residual per (locality, year), MSD §2.8.7 "open national residual".

    ``NetMig(s,t) = E(s,t) - E(s,t-1) - Births(s,t) + Deaths(s,t)`` — the demographic
    balancing equation solved for the unobserved term. Computed only for CONSECUTIVE
    calendar years (both closure totals present); over a multi-year gap the residual
    would be a cumulative, not annual, flow and is left unobserved. Absent birth/death
    keys are read as zero registered events (the total-only SIDRA cell semantics),
    which is why civil-registry facts — where every requested locality-year returns a
    row — are the preferred source over sparser DATASUS counts.
    """
    if births_by_st is None and deaths_by_st is None:
        return None, []
    births_by_st = births_by_st or {}
    deaths_by_st = deaths_by_st or {}
    s_count, t_count = shape[0], shape[1]
    values: list[float | None] = [None] * (s_count * t_count)
    for s, locality in enumerate(localities):
        for t in range(1, t_count):
            if int(periods[t]) - int(periods[t - 1]) != 1:
                continue
            e_now = closure[s * t_count + t]
            e_prev = closure[s * t_count + (t - 1)]
            if e_now is None or e_prev is None:
                continue
            births = births_by_st.get((locality, periods[t]), 0.0)
            deaths = deaths_by_st.get((locality, periods[t]), 0.0)
            values[s * t_count + t] = float(e_now) - float(e_prev) - births + deaths
    if not any(value is not None for value in values):
        return None, []
    return tuple(values), []


def _reanchor_closure_single_vintage(
    *,
    closure: list[float | None],
    localities: tuple[str, ...],
    periods: tuple[str, ...],
    census_years: frozenset[str],
    shape: tuple[int, ...],
    period_index: dict[str, int],
) -> dict[str, int]:
    """Overwrite intercensal closure cells with census-anchored geometric interpolation, in place.

    See the FAL-POP-SV block in :func:`solve_population_tensor_from_sidra_strata`. Returns telemetry:
    how many municipalities were re-anchored vs kept on their prior (6579) closure for lack of >=2
    census anchors, and the count of cells rewritten.
    """
    n_periods = shape[1]
    census_period_list = sorted((p for p in periods if p in census_years), key=lambda p: int(p))
    target_year_ints = [int(p) for p in periods]
    reanchored = 0
    kept_single_anchor = 0
    cells_rewritten = 0
    for s in range(len(localities)):
        anchors: dict[int, float] = {}
        for cp in census_period_list:
            value = closure[s * n_periods + period_index[cp]]
            if value is not None and value > 0.0:
                anchors[int(cp)] = float(value)
        if len(anchors) < 2:
            kept_single_anchor += 1
            continue
        interpolated = geometric_interpolate_closure(anchors, target_year_ints)
        for p in periods:
            year = int(p)
            if year in anchors:
                continue  # census cell stays on its own vintage
            new_value = interpolated.get(year)
            if new_value is None:
                continue
            closure[s * n_periods + period_index[p]] = float(new_value)
            cells_rewritten += 1
        reanchored += 1
    return {
        "reanchored_municipalities": reanchored,
        "kept_single_anchor_municipalities": kept_single_anchor,
        "cells_rewritten": cells_rewritten,
    }


__all__ = [
    "_sidra_vital_totals",
    "_datasus_event_totals",
    "_migration_residual_totals",
    "_reanchor_closure_single_vintage",
]
