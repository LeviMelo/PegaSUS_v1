"""DATASUS-origin flow priors (SIM deaths, SINASC births) + census race-composition prior (MSD §2.8).

Administrative race (SIM/SINASC) enters a real self-declared race axis only through the RaceBridge
(§2.8.5/§2.8.6); the census race-composition prior instead uses this tensor's own self-declared 9606
counts (§2.8.8) and needs no bridge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pegasus.measurement.race import RaceBridgePrior, bridge_admin_race_group_counts
from pegasus.registries.demographic_axis import TOTAL, age_group_for_years

from pegasus.denominators.population.build.indexing import *  # noqa: F401,F403 (intra-package base layer)
from pegasus.denominators.population.build.indexing import (
    _birth_cell_index,
    _cell_index,
    _census_count_arrays,
)
from pegasus.denominators.population.build.layer1 import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.layer1 import _interpolate_shares


def _resolve_geo_year_columns(frame: pl.DataFrame, *, geo_candidates: tuple[str, ...], year_candidates: tuple[str, ...]) -> pl.DataFrame | None:
    if "municipality_cod6" not in frame.columns:
        geo_column = next((column for column in geo_candidates if column in frame.columns), None)
        if geo_column is None:
            return None
        frame = frame.with_columns(pl.col(geo_column).cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("municipality_cod6"))
    if "year" not in frame.columns:
        year_column = next((column for column in year_candidates if column in frame.columns), None)
        if year_column is None:
            return None
        frame = frame.with_columns(pl.col(year_column).cast(pl.Int64, strict=False).alias("year"))
    if "municipality_cod6" not in frame.columns or "year" not in frame.columns:
        return None
    return frame


def _stratify_sex_column(frame: pl.DataFrame, *, source_system: str, sex_column: str, sex_index: dict[str, int]) -> pl.DataFrame:
    from pegasus.registries.demographic_axis import source_category_map

    sex_map = source_category_map("sex", source_system)
    if sex_column in frame.columns and any(value != TOTAL for value in sex_index) and sex_map:
        return frame.with_columns(
            pl.col(sex_column).cast(pl.Utf8, strict=False).replace(sex_map).alias("__sex__")
        ).filter(pl.col("__sex__").is_in(list(sex_index)))
    return frame.with_columns(pl.lit(TOTAL).alias("__sex__"))


def _stratify_age_column(frame: pl.DataFrame, *, age_column: str, age_index: dict[str, int]) -> pl.DataFrame:
    if age_column in frame.columns and any(value != TOTAL for value in age_index):
        return frame.with_columns(
            pl.col(age_column).map_elements(age_group_for_years, return_dtype=pl.Utf8).alias("__age_group__")
        ).filter(pl.col("__age_group__").is_in(list(age_index)))
    return frame.with_columns(pl.lit(TOTAL).alias("__age_group__"))


def _coalesce_race_columns(
    frame: pl.DataFrame,
    *,
    primary_code: str,
    primary_state: str,
    fallback_code: str,
    fallback_state: str,
    out_code: str = "__race_code__",
    out_state: str = "__race_state__",
) -> pl.DataFrame:
    """Prefer the primary administrative race, fall back to a secondary one.

    MSD §2.8.5 orders newborn-race imputation ``P(r_n | r_m, s) -> ...`` when
    newborn race is unavailable. SINASC carries BOTH ``newborn_race_admin`` and
    ``maternal_race_admin``; when the newborn's own race is missing/invalid we take
    the mother's declared race as a first-order stand-in for that conditional
    (``r_n := r_m``), which is the leading term of the §2.8.5 cascade — the full
    P(r_n|r_m,s) allocation table is not yet estimated. Both are administrative
    codes and both go through Bridge_R downstream; this only decides which raw code
    feeds it per birth. Rows valid on neither race are left for the missing/local-pi
    reallocation inside Bridge_R.
    """
    has_primary = primary_code in frame.columns
    has_fallback = fallback_code in frame.columns
    if not has_primary and not has_fallback:
        return frame.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias(out_code),
            pl.lit(None, dtype=pl.Utf8).alias(out_state),
        )
    primary_valid = (
        (pl.col(primary_state).is_in(["valid_admin_race", "valid"])) if (has_primary and primary_state in frame.columns)
        else pl.col(primary_code).is_not_null() if has_primary
        else pl.lit(False)
    )
    primary_code_expr = pl.col(primary_code) if has_primary else pl.lit(None, dtype=pl.Utf8)
    primary_state_expr = pl.col(primary_state) if (has_primary and primary_state in frame.columns) else pl.lit(None, dtype=pl.Utf8)
    fallback_code_expr = pl.col(fallback_code) if has_fallback else pl.lit(None, dtype=pl.Utf8)
    fallback_state_expr = pl.col(fallback_state) if (has_fallback and fallback_state in frame.columns) else pl.lit(None, dtype=pl.Utf8)
    return frame.with_columns(
        pl.when(primary_valid).then(primary_code_expr).otherwise(fallback_code_expr).alias(out_code),
        pl.when(primary_valid).then(primary_state_expr).otherwise(fallback_state_expr).alias(out_state),
    )


def _bridge_race_stratified_counts(
    frame: pl.DataFrame,
    *,
    race_column: str,
    race_state_column: str | None,
    group_keys: list[str],
    race_bridge_prior: RaceBridgePrior | None,
    race_index: dict[str, int],
    warning_code: str,
) -> tuple[pl.DataFrame | None, str | None]:
    """Group ``frame`` and, when a real race axis is modeled, bridge each group's
    raw administrative race codes to the self-declared race categories (MSD
    §3.7.4 / §2.8.5-§2.8.6). Returns ``(grouped_with_race_and_count, warning)``
    where the frame has one row per (group_keys..., race, count); ``None`` when
    the axis cannot be honestly populated (real race axis, no bridge prior).
    """
    if not any(value != TOTAL for value in race_index):
        grouped = frame.group_by(group_keys).agg(pl.len().cast(pl.Float64).alias("__count__"))
        return grouped.with_columns(pl.lit(TOTAL).alias("__race__")), None
    if race_bridge_prior is None or race_column not in frame.columns:
        return None, warning_code
    agg_exprs = [pl.col(race_column).alias("__race_codes__")]
    if race_state_column and race_state_column in frame.columns:
        agg_exprs.append(pl.col(race_state_column).alias("__race_states__"))
    grouped = frame.group_by(group_keys).agg(*agg_exprs)
    rows: list[dict[str, Any]] = []
    for row in grouped.iter_rows(named=True):
        posterior = bridge_admin_race_group_counts(
            race_codes=row["__race_codes__"],
            race_states=row.get("__race_states__"),
            prior=race_bridge_prior,
            support={key: row[key] for key in group_keys},
        )
        for race, count in posterior.posterior_counts.items():
            if race not in race_index:
                continue
            rows.append({**{key: row[key] for key in group_keys}, "__race__": race, "__count__": float(count)})
    if not rows:
        return pl.DataFrame(schema={**{key: frame.schema[key] for key in group_keys}, "__race__": pl.Utf8, "__count__": pl.Float64}), None
    return pl.DataFrame(rows), None


def _sim_death_priors(
    *,
    sim_events_path: str | Path | None,
    locality_index: dict[str, int],
    period_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    shape: tuple[int, int, int, int, int],
    race_bridge_prior: RaceBridgePrior | None = None,
) -> tuple[tuple[float | None, ...] | None, list[str]]:
    if sim_events_path is None:
        return None, []
    path = Path(sim_events_path)
    if not path.exists():
        return None, []
    frame = _resolve_geo_year_columns(
        pl.read_parquet(path),
        geo_candidates=("mun_residence_cod6", "mun_occurrence_cod6", "CODMUNRES", "MUNIC_RES"),
        year_candidates=("death_year", "event_year", "ANO"),
    )
    if frame is None:
        return None, []
    frame = _stratify_sex_column(frame, source_system="SIM-DO", sex_column="sex", sex_index=sex_index)
    frame = _stratify_age_column(frame, age_column="age_years", age_index=age_index)
    grouped, warning = _bridge_race_stratified_counts(
        frame,
        race_column="race_color_admin",
        race_state_column="race_missingness_state",
        group_keys=["municipality_cod6", "year", "__sex__", "__age_group__"],
        race_bridge_prior=race_bridge_prior,
        race_index=race_index,
        warning_code="sim_death_race_stratification_unavailable_without_bridge",
    )
    if grouped is None:
        return None, [warning] if warning else []

    # Vectorized scatter (§V.1): the flat cell index is computed for all grouped rows at once and
    # scatter-added, so the O(n_cells) death prior is a single float64 array (NaN = absent, was None)
    # instead of a ~1 GB Python None-list filled by per-row _cell_index calls. The flat-index
    # arithmetic ((((s*T)+t)*A+a)*X+x)*R+r reproduces _cell_index's (s,t,a,x,r) layout exactly; the
    # zero-init + np.add.at accumulation matches the old ``(values[idx] or 0.0) + count``; unset cells
    # become NaN, the absent sentinel every consumer already treats identically to the old None.
    s_count, t_count, a_count, x_count, r_count = shape
    n_cells = s_count * t_count * a_count * x_count * r_count
    n_rows = grouped.height
    li = np.fromiter((locality_index.get(str(v), -1) for v in grouped["municipality_cod6"]), dtype=np.int64, count=n_rows)
    pi = np.fromiter((period_index.get(str(v), -1) for v in grouped["year"]), dtype=np.int64, count=n_rows)
    ai = np.fromiter((age_index.get(str(v), -1) for v in grouped["__age_group__"]), dtype=np.int64, count=n_rows)
    xi = np.fromiter((sex_index.get(str(v), -1) for v in grouped["__sex__"]), dtype=np.int64, count=n_rows)
    ri = np.fromiter((race_index.get(str(v), -1) for v in grouped["__race__"]), dtype=np.int64, count=n_rows)
    counts = np.fromiter((float(v) for v in grouped["__count__"]), dtype=np.float64, count=n_rows)
    valid = (li >= 0) & (pi >= 0) & (ai >= 0) & (xi >= 0) & (ri >= 0)
    flat_idx = (((li * t_count + pi) * a_count + ai) * x_count + xi) * r_count + ri
    values = np.zeros(n_cells, dtype=np.float64)
    np.add.at(values, flat_idx[valid], counts[valid])
    touched = np.zeros(n_cells, dtype=bool)
    touched[flat_idx[valid]] = True
    if not touched.any():
        return None, []
    values[~touched] = np.nan  # NaN = no death prior for that cell (was None)
    return values, []


def _sinasc_birth_priors(
    *,
    sinasc_events_path: str | Path | None,
    locality_index: dict[str, int],
    period_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    t_count: int,
    x_count: int,
    r_count: int,
    race_bridge_prior: RaceBridgePrior | None = None,
) -> tuple[tuple[float | None, ...] | None, list[str]]:
    """SINASC newborn birth counts feeding the birth loss's ``B^newborn`` term
    (MSD §2.8.5): population entry at age 0, stratified by (locality, year, sex,
    race). Newborn race is administrative (``newborn_race_admin``) and, like SIM
    death race, requires a RaceBridge prior to enter a real race axis."""
    if sinasc_events_path is None:
        return None, []
    path = Path(sinasc_events_path)
    if not path.exists():
        return None, []
    frame = _resolve_geo_year_columns(
        pl.read_parquet(path),
        geo_candidates=("mun_residence_cod6", "CODMUNRES", "MUNIC_RES"),
        year_candidates=("birth_year", "event_year", "DTNASC"),
    )
    if frame is None:
        return None, []
    frame = _stratify_sex_column(frame, source_system="SINASC", sex_column="newborn_sex", sex_index=sex_index)
    # SINASC exposes BOTH newborn and maternal administrative race; prefer the
    # newborn's own declaration, fall back to the mother's (MSD §2.8.5 r_n|r_m).
    frame = _coalesce_race_columns(
        frame,
        primary_code="newborn_race_admin",
        primary_state="newborn_race_state",
        fallback_code="maternal_race_admin",
        fallback_state="maternal_race_state",
    )
    grouped, warning = _bridge_race_stratified_counts(
        frame,
        race_column="__race_code__",
        race_state_column="__race_state__",
        group_keys=["municipality_cod6", "year", "__sex__"],
        race_bridge_prior=race_bridge_prior,
        race_index=race_index,
        warning_code="sinasc_birth_race_stratification_unavailable_without_bridge",
    )
    if grouped is None:
        return None, [warning] if warning else []

    # Vectorized scatter (§V.1): the birth support has no age axis, so the flat index is
    # ((s*T+t)*X+x)*R+r -- _birth_cell_index's (s,t,x,r) layout, computed for all grouped rows at
    # once and scatter-added into one float64 array (NaN = absent, was None). Same zero-init +
    # np.add.at accumulation as the old ``(values[idx] or 0.0) + count``.
    s_count = len(locality_index)
    n = s_count * t_count * x_count * r_count
    n_rows = grouped.height
    li = np.fromiter((locality_index.get(str(v), -1) for v in grouped["municipality_cod6"]), dtype=np.int64, count=n_rows)
    pi = np.fromiter((period_index.get(str(v), -1) for v in grouped["year"]), dtype=np.int64, count=n_rows)
    xi = np.fromiter((sex_index.get(str(v), -1) for v in grouped["__sex__"]), dtype=np.int64, count=n_rows)
    ri = np.fromiter((race_index.get(str(v), -1) for v in grouped["__race__"]), dtype=np.int64, count=n_rows)
    counts = np.fromiter((float(v) for v in grouped["__count__"]), dtype=np.float64, count=n_rows)
    valid = (li >= 0) & (pi >= 0) & (xi >= 0) & (ri >= 0)
    flat_idx = ((li * t_count + pi) * x_count + xi) * r_count + ri
    values = np.zeros(n, dtype=np.float64)
    np.add.at(values, flat_idx[valid], counts[valid])
    touched = np.zeros(n, dtype=bool)
    touched[flat_idx[valid]] = True
    if not touched.any():
        return None, []
    values[~touched] = np.nan  # NaN = no birth prior for that cell (was None)
    return values, []


def _census_race_composition_prior(
    *,
    records: list[dict[str, Any]],
    locality_index: dict[str, int],
    period_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    shape: tuple[int, int, int, int, int],
) -> tuple[float | None, ...] | None:
    """Race composition prior for the race loss (MSD §2.8.8).

    Built purely from this tensor's own SIDRA 9606 self-declared race counts
    (``records``) -- unlike the death/birth priors this needs no RaceBridge:
    §2.8.8's ``z^bridge`` is an ILR interpolation between two census self-declared
    compositions, not a DATASUS-origin administrative crosswalk. Every (locality,
    age, sex) cell's race distribution is computed at each census year present in
    ``records``, then linearly interpolated (in proportion space) across the full
    time axis between the bracketing census years. A run year outside the
    observed census bracket is clamped to the nearest census composition -- MSD
    §2.8.8 only defines the interpolated case, so this boundary extension is a
    deliberate, conservative choice (a flat carry-forward/back of the last known
    composition), not part of the formula itself.
    """
    if not any(value != TOTAL for value in race_index) or len(race_index) < 2:
        return None
    s_count, t_count, a_count, x_count, r_count = shape
    group_size = a_count * x_count * r_count
    # Vectorized (§V.1): per (locality, census year) build the [group_size] count vector, normalize
    # over RACE within each (age,sex) block to a race composition, interpolate across periods with
    # numpy, and write a contiguous per-(locality, period) slice. NaN marks cells with no census race
    # data (masked out by the loss, exactly as the old per-cell None did). No per-cell Python loop.
    census_counts = _census_count_arrays(
        records, locality_index=locality_index, age_index=age_index, sex_index=sex_index,
        race_index=race_index, group_size=group_size, x_count=x_count, r_count=r_count,
    )
    n = s_count * t_count * group_size
    values = np.full(n, np.nan, dtype=np.float64)
    period_pairs = sorted(period_index.items(), key=lambda kv: int(kv[0]))
    any_set = False
    for li, by_year in census_counts.items():
        race_shares: dict[str, np.ndarray] = {}
        for year, arr in by_year.items():
            block = arr.reshape(a_count, x_count, r_count)
            block_sum = block.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(block_sum > 0, block / block_sum, np.nan)  # NaN where the block has no data
            race_shares[year] = share.reshape(group_size)
        census_years = sorted(race_shares, key=int)
        if not census_years:
            continue
        for period, pi in period_pairs:
            interp = _interpolate_shares(race_shares, census_years, int(period))
            base = (li * t_count + pi) * group_size
            values[base:base + group_size] = interp
            any_set = True
    if not any_set or bool(np.isnan(values).all()):
        return None
    return values


__all__ = [
    "_resolve_geo_year_columns",
    "_stratify_sex_column",
    "_stratify_age_column",
    "_coalesce_race_columns",
    "_bridge_race_stratified_counts",
    "_sim_death_priors",
    "_sinasc_birth_priors",
    "_census_race_composition_prior",
]
