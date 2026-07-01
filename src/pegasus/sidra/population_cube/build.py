"""Population tensor solver orchestration (MSD §2.8).

This module turns real disaggregated SIDRA 9606 facts into a
``PopulationTensorProblem`` over ``(municipality, year, age_group, sex, race)`` and
runs the existing analytic solver. It is intentionally projection-aware: only source
categories that can be mapped through ``demographic/demographic_axis_maps.yaml`` become strata.
Unmapped or unknown categories are excluded rather than silently relabelled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.efg.race_bridge import RaceBridgePrior, bridge_admin_race_group_counts, load_race_bridge_prior
from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, age_group_for_years, age_group_sort_key, map_category
from pegasus.registries.population import assert_dense_population_tensor_allowed, select_population_solver
from pegasus.she.reconstruction.diagnostics import population_tensor_diagnostics
from pegasus.she.reconstruction.schema import (
    PopulationObjectiveWeights,
    PopulationTensorProblem,
    PopulationTensorRequest,
    PopulationTensorResult,
)
from pegasus.she.reconstruction.solvers import solve_population_tensor_problem
from pegasus.sidra.population_cube.anchor import load_combined_population_totals_frame, load_sidra_population_total_anchor


AXES = ("age_group", "sex", "race")
AXIS_CLASSIFICATIONS = {"sex": "2", "race": "86", "age_group": "287"}

# SIDRA civil-registry vital-statistics tables (IBGE Registro Civil), the
# internally-consistent source for the net-migration residual (MSD §2.8.7):
# NetMig(s,t) = E(s,t) - E(s,t-1) - Births(s,t) + Deaths(s,t). Same IBGE universe
# as the 6579 population estimates, unlike DATASUS SIM/SINASC (used as fallback).
SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE = "2609"
SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE = "217"
SIDRA_CIVIL_REGISTRY_DEATHS_TABLE = "2683"
SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE = "343"

# MSD §2.8.5/§2.8.6: administrative race/color (SIM race_color_admin, SINASC
# newborn_race_admin) is declaration-process incompatible with this tensor's
# self-declared IBGE race axis (§3.7.4) -- it may only enter a race-stratified
# cell through pegasus.efg.race_bridge, never a direct category crosswalk.
# When the race axis is real (not the degenerate TOTAL-only case) and no bridge
# prior was supplied, DATASUS-origin priors are left out of that axis entirely
# rather than silently collapsed onto an unmodeled TOTAL cell.


@dataclass(frozen=True)
class PopulationTensorBuild:
    result: PopulationTensorResult
    output_path: str
    locality_ids: tuple[str, ...]
    periods: tuple[str, ...]
    age_groups: tuple[str, ...]
    sexes: tuple[str, ...]
    races: tuple[str, ...]
    migration_flows_path: str | None = None
    migration_affinity_path: str | None = None
    migration_flow_manifests: tuple[dict[str, Any], ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            **self.result.as_manifest(),
            "output_path": self.output_path,
            "locality_ids": list(self.locality_ids),
            "periods": list(self.periods),
            "age_groups": list(self.age_groups),
            "sexes": list(self.sexes),
            "races": list(self.races),
            "migration_flows_path": self.migration_flows_path,
            "migration_affinity_path": self.migration_affinity_path,
            "migration_flow_reconstructions": list(self.migration_flow_manifests),
        }


def _pairs(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    return [(str(item[0]), str(item[1])) for item in (parsed or []) if len(item) >= 2]


def _category_by_classification(raw: Any) -> dict[str, str]:
    return {classification: category for classification, category in _pairs(raw)}


def _canonical_stratum(row: dict[str, Any]) -> dict[str, str] | None:
    categories = _category_by_classification(row.get("category_tuple"))
    out: dict[str, str] = {}
    for axis, classification_id in AXIS_CLASSIFICATIONS.items():
        raw_code = categories.get(classification_id)
        if raw_code is None:
            out[axis] = TOTAL
            continue
        canonical = map_category(axis, "SIDRA", raw_code)
        if canonical == UNKNOWN:
            return None
        out[axis] = canonical
    return out


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


def _read_population_strata(path: str | Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    required = {"table_id", "variable_id", "period", "locality_id", "category_tuple", "value_numeric", "value_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"population strata facts missing columns: {sorted(missing)}")
    return frame.filter(
        (pl.col("table_id").cast(pl.Utf8) == "9606")
        & (pl.col("variable_id").cast(pl.Utf8) == "93")
        & (pl.col("value_status").cast(pl.Utf8) == "numeric")
        & pl.col("value_numeric").is_not_null()
    )


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
        (pl.col(primary_state) == "valid_admin_race") if (has_primary and primary_state in frame.columns)
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

    values: list[float | None] = [None] * (shape[0] * shape[1] * shape[2] * shape[3] * shape[4])
    for row in grouped.iter_rows(named=True):
        locality = str(row["municipality_cod6"])
        period = str(row["year"])
        sex = str(row["__sex__"])
        age_group = str(row["__age_group__"])
        race = str(row["__race__"])
        if (
            locality not in locality_index
            or period not in period_index
            or sex not in sex_index
            or age_group not in age_index
            or race not in race_index
        ):
            continue
        idx = _cell_index(
            locality=locality,
            period=period,
            age_group=age_group,
            sex=sex,
            race=race,
            locality_index=locality_index,
            period_index=period_index,
            age_index=age_index,
            sex_index=sex_index,
            race_index=race_index,
            shape=shape,
        )
        values[idx] = (values[idx] or 0.0) + float(row["__count__"])
    if not any(value is not None for value in values):
        return None, []
    return tuple(values), []


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

    n = len(locality_index) * t_count * x_count * r_count
    values: list[float | None] = [None] * n
    for row in grouped.iter_rows(named=True):
        locality = str(row["municipality_cod6"])
        period = str(row["year"])
        sex = str(row["__sex__"])
        race = str(row["__race__"])
        if locality not in locality_index or period not in period_index or sex not in sex_index or race not in race_index:
            continue
        idx = _birth_cell_index(
            locality=locality,
            period=period,
            sex=sex,
            race=race,
            locality_index=locality_index,
            period_index=period_index,
            sex_index=sex_index,
            race_index=race_index,
            t_count=t_count,
            x_count=x_count,
            r_count=r_count,
        )
        values[idx] = (values[idx] or 0.0) + float(row["__count__"])
    if not any(value is not None for value in values):
        return None, []
    return tuple(values), []


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
    # cell_composition[(locality, age, sex)][period] = {race: proportion}
    totals: dict[tuple[str, str, str], dict[str, float]] = {}
    for record in records:
        key = (record["municipality_cod6"], record["age_group"], record["sex"])
        totals.setdefault(key, {}).setdefault(record["period"], {})
        totals[key][record["period"]][record["race"]] = totals[key].get(record["period"], {}).get(record["race"], 0.0) + float(record["value"])
    census_years_sorted = sorted({record["period"] for record in records})
    if not census_years_sorted:
        return None

    n = shape[0] * shape[1] * shape[2] * shape[3] * shape[4]
    values: list[float | None] = [None] * n
    for (locality, age_group, sex), by_year in totals.items():
        if locality not in locality_index or age_group not in age_index or sex not in sex_index:
            continue
        proportions: dict[str, dict[str, float]] = {}
        for year, race_counts in by_year.items():
            total = sum(race_counts.values())
            if total <= 0:
                continue
            proportions[year] = {race: count / total for race, count in race_counts.items()}
        years_available = sorted(proportions)
        if not years_available:
            continue
        for period in period_index:
            year_int = int(period)
            anchors_int = [int(year) for year in years_available]
            if year_int <= anchors_int[0]:
                lo = hi = years_available[0]
                weight = 0.0
            elif year_int >= anchors_int[-1]:
                lo = hi = years_available[-1]
                weight = 0.0
            else:
                lo = years_available[max(i for i, y in enumerate(anchors_int) if y <= year_int)]
                hi = years_available[min(i for i, y in enumerate(anchors_int) if y >= year_int)]
                weight = 0.0 if lo == hi else (year_int - int(lo)) / (int(hi) - int(lo))
            lo_dist, hi_dist = proportions[lo], proportions[hi]
            for race in race_index:
                p = (1.0 - weight) * lo_dist.get(race, 0.0) + weight * hi_dist.get(race, 0.0)
                idx = _cell_index(
                    locality=locality,
                    period=period,
                    age_group=age_group,
                    sex=sex,
                    race=race,
                    locality_index=locality_index,
                    period_index=period_index,
                    age_index=age_index,
                    sex_index=sex_index,
                    race_index=race_index,
                    shape=shape,
                )
                values[idx] = p
    if not any(value is not None for value in values):
        return None
    return tuple(values)


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


def _reconstruct_and_persist_migration_flows(
    *,
    localities: tuple[str, ...],
    periods: tuple[str, ...],
    closure: list[float | None],
    migration_locality_totals: tuple[float | None, ...] | None,
    shape: tuple[int, int, int, int, int],
    out_path: Path,
    contiguity_graph_id: str,
    max_hops: int,
) -> tuple[str | None, str | None, tuple[dict[str, Any], ...]]:
    """Reconstruct O→D migration flows + affinity kernel from the tensor's own net
    residual and closure populations (MSD §2.8.7 flow layer). Returns
    ``(flows_path, affinity_path, per_year_manifests)``; a no-op ``(None, None, ())``
    when there is no net-migration signal or the contiguity graph is unavailable."""
    if migration_locality_totals is None:
        return None, None, ()
    from pegasus.geo.migration_affinity import build_migration_affinity_graph
    from pegasus.geo.spatial_graph import structural_cod6_adjacency
    from pegasus.sidra.population_cube.migration import MigrationFlowError, reconstruct_migration_flows

    try:
        full_adjacency = structural_cod6_adjacency(contiguity_graph_id)
    except Exception:
        return None, None, ()
    locality_set = set(localities)
    adjacency = {loc: tuple(n for n in full_adjacency.get(loc, ()) if n in locality_set) for loc in localities}

    t_count = shape[1]
    populations_by_year: dict[str, dict[str, float]] = {}
    net_by_year: dict[str, dict[str, float]] = {}
    for t, period in enumerate(periods):
        pops: dict[str, float] = {}
        nets: dict[str, float] = {}
        for s, loc in enumerate(localities):
            c = closure[s * t_count + t]
            if c is not None:
                pops[loc] = float(c)
            m = migration_locality_totals[s * t_count + t]
            if m is not None:
                nets[loc] = float(m)
        populations_by_year[period] = pops
        net_by_year[period] = nets

    try:
        reconstructions = reconstruct_migration_flows(
            nodes=list(localities),
            populations_by_year=populations_by_year,
            net_by_year=net_by_year,
            adjacency=adjacency,
            max_hops=max_hops,
        )
    except MigrationFlowError:
        return None, None, ()
    if not reconstructions:
        return None, None, ()

    flow_rows = [
        {"year": int(rec.year), "origin_cod6": i, "destination_cod6": j, "flow": value}
        for rec in reconstructions for (i, j), value in rec.flows.items()
    ]
    flows_path = out_path.with_suffix(".migration_flows.parquet")
    pl.DataFrame(flow_rows, schema={"year": pl.Int64, "origin_cod6": pl.Utf8, "destination_cod6": pl.Utf8, "flow": pl.Float64}).write_parquet(flows_path)

    # Representative population per node (mean over years present) for the mass
    # normalization of the affinity kernel.
    pop_accum: dict[str, list[float]] = {}
    for pops in populations_by_year.values():
        for loc, value in pops.items():
            pop_accum.setdefault(loc, []).append(value)
    populations_by_node = {loc: (sum(v) / len(v)) for loc, v in pop_accum.items() if v}
    graph = build_migration_affinity_graph(reconstructions, populations_by_node)
    edge_rows: list[dict[str, Any]] = []
    weight_matrix = graph.view("weight")
    index = graph.index
    for i in graph.node_ids:
        for j in graph.neighbors(i):
            if i < j:  # undirected: emit each edge once
                edge_rows.append({"source_cod6": i, "target_cod6": j, "affinity": float(weight_matrix[index[i], index[j]])})
    affinity_path = out_path.with_suffix(".migration_affinity.parquet")
    pl.DataFrame(edge_rows, schema={"source_cod6": pl.Utf8, "target_cod6": pl.Utf8, "affinity": pl.Float64}).write_parquet(affinity_path)
    return str(flows_path), str(affinity_path), tuple(rec.as_manifest() for rec in reconstructions)


def solve_population_tensor_from_sidra_strata(
    *,
    population_strata_path: str | Path,
    total_anchor_path: str | Path,
    output_path: str | Path,
    mode: str = "independent_denominator",
    sim_events_path: str | Path | None = None,
    sinasc_events_path: str | Path | None = None,
    civil_registry_births_path: str | Path | None = None,
    civil_registry_deaths_path: str | Path | None = None,
    race_bridge_prior_path: str | Path | None = None,
    reconstruct_migration: bool = False,
    contiguity_graph_id: str = "contiguity_queen",
    migration_max_hops: int = 3,
    solver_id: str | None = None,
    max_iterations: int = 2_000,
    tolerance: float = 1e-5,
) -> PopulationTensorBuild:
    strata_path = Path(population_strata_path)
    totals_path = Path(total_anchor_path)
    out_path = Path(output_path)
    if mode not in {"independent_denominator", "sim_informed_denominator"}:
        raise ValueError(f"Unsupported population tensor solver mode: {mode}")
    race_bridge_prior = load_race_bridge_prior(race_bridge_prior_path) if race_bridge_prior_path is not None else None

    strata = _read_population_strata(strata_path)
    if strata.height == 0:
        raise ValueError("population strata artifact contains no numeric SIDRA 9606 population rows")

    records: list[dict[str, Any]] = []
    categories_by_axis: dict[str, set[str]] = {axis: set() for axis in AXES}
    for row in strata.iter_rows(named=True):
        stratum = _canonical_stratum(row)
        if stratum is None:
            continue
        record = {
            "municipality_cod6": str(row["locality_id"])[:6],
            "period": str(row["period"])[:4],
            "value": float(row["value_numeric"]),
            **stratum,
        }
        records.append(record)
        for axis in AXES:
            categories_by_axis[axis].add(record[axis])
    if not records:
        raise ValueError("population strata artifact has no registry-projectable demographic cells")

    # `totals` stitches the census-year (9606) and intercensal (6579) population
    # totals into one (municipality, year) closure panel (MSD §2.8.10); its period
    # coverage is a superset of the strata's (strata/disaggregation only exists for
    # census years) -- the tensor's time axis must span BOTH so intercensal years
    # get a real closure anchor instead of silently having none.
    totals = load_combined_population_totals_frame(totals_path)
    localities = tuple(sorted({record["municipality_cod6"] for record in records} | {str(row) for row in totals["municipality_cod6"].to_list()}))
    periods = tuple(sorted({record["period"] for record in records} | {str(int(row)) for row in totals["year"].to_list()}))
    age_groups = tuple(sorted(categories_by_axis["age_group"] or {TOTAL}, key=age_group_sort_key))
    sexes = tuple(sorted(categories_by_axis["sex"] or {TOTAL}))
    races = tuple(sorted(categories_by_axis["race"] or {TOTAL}))
    shape = (len(localities), len(periods), len(age_groups), len(sexes), len(races))

    locality_index = {value: idx for idx, value in enumerate(localities)}
    period_index = {value: idx for idx, value in enumerate(periods)}
    age_index = {value: idx for idx, value in enumerate(age_groups)}
    sex_index = {value: idx for idx, value in enumerate(sexes)}
    race_index = {value: idx for idx, value in enumerate(races)}

    n_cells = shape[0] * shape[1] * shape[2] * shape[3] * shape[4]
    anchors: list[float | None] = [None] * n_cells
    for record in records:
        idx = _cell_index(
            locality=record["municipality_cod6"],
            period=record["period"],
            age_group=record["age_group"],
            sex=record["sex"],
            race=record["race"],
            locality_index=locality_index,
            period_index=period_index,
            age_index=age_index,
            sex_index=sex_index,
            race_index=race_index,
            shape=shape,
        )
        anchors[idx] = (anchors[idx] or 0.0) + float(record["value"])

    closure: list[float | None] = [None] * (shape[0] * shape[1])
    for row in totals.iter_rows(named=True):
        locality = str(row["municipality_cod6"])
        period = str(row["year"])
        if locality in locality_index and period in period_index:
            closure[locality_index[locality] * shape[1] + period_index[period]] = float(row["value"])
    for s in range(shape[0]):
        for t in range(shape[1]):
            idx = s * shape[1] + t
            if closure[idx] is not None:
                continue
            start = ((s * shape[1] + t) * shape[2] * shape[3] * shape[4])
            end = start + shape[2] * shape[3] * shape[4]
            observed = [value for value in anchors[start:end] if value is not None]
            closure[idx] = float(sum(observed)) if observed else None

    sim_deaths, death_warnings = _sim_death_priors(
        sim_events_path=sim_events_path,
        locality_index=locality_index,
        period_index=period_index,
        age_index=age_index,
        sex_index=sex_index,
        race_index=race_index,
        shape=shape,
        race_bridge_prior=race_bridge_prior,
    )
    births, birth_warnings = _sinasc_birth_priors(
        sinasc_events_path=sinasc_events_path,
        locality_index=locality_index,
        period_index=period_index,
        sex_index=sex_index,
        race_index=race_index,
        t_count=shape[1],
        x_count=shape[3],
        r_count=shape[4],
        race_bridge_prior=race_bridge_prior,
    )
    race_composition_prior = _census_race_composition_prior(
        records=records,
        locality_index=locality_index,
        period_index=period_index,
        age_index=age_index,
        sex_index=sex_index,
        race_index=race_index,
        shape=shape,
    )
    # Net-migration residual (MSD §2.8.7): SIDRA civil-registry vital totals preferred
    # (IBGE-universe-consistent with the population estimates), DATASUS SIM/SINASC
    # counts as fallback when no civil-registry facts were acquired.
    births_by_st = _sidra_vital_totals(
        civil_registry_births_path, table_id=SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE, variable_id=SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE
    )
    if births_by_st is None:
        births_by_st = _datasus_event_totals(
            sinasc_events_path, geo_candidates=("mun_residence_cod6", "CODMUNRES", "MUNIC_RES"), year_candidates=("birth_year", "event_year")
        )
    deaths_by_st = _sidra_vital_totals(
        civil_registry_deaths_path, table_id=SIDRA_CIVIL_REGISTRY_DEATHS_TABLE, variable_id=SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE
    )
    if deaths_by_st is None:
        deaths_by_st = _datasus_event_totals(
            sim_events_path, geo_candidates=("mun_residence_cod6", "mun_occurrence_cod6", "CODMUNRES", "MUNIC_RES"), year_candidates=("death_year", "event_year")
        )
    migration_locality_totals, migration_warnings = _migration_residual_totals(
        closure=closure,
        births_by_st=births_by_st,
        deaths_by_st=deaths_by_st,
        localities=localities,
        periods=periods,
        shape=shape,
    )
    # Per-cell migration bound. MSD §2.8.1 bounds eta by the closure total E_{s,t},
    # which exists for EVERY year (census + intercensal); the per-cell strata anchor
    # does not (only census years), so basing the bound on it starves intercensal
    # years of migration headroom -- exactly where the residual is most needed.
    strata_per_st = shape[2] * shape[3] * shape[4]
    migration_bounds: list[float] = []
    for idx in range(n_cells):
        closure_total = closure[idx // strata_per_st]
        if closure_total is not None and closure_total > 0:
            migration_bounds.append(max(0.25 * float(closure_total) / strata_per_st, 1.0))
        else:
            migration_bounds.append(max((anchors[idx] or 0.0) * 0.25, 1.0))

    death_rates: tuple[float | None, ...] | None = None
    warnings: list[str] = [*death_warnings, *birth_warnings, *migration_warnings]
    feedback_warning = False
    reconstruction_uncertainty = 0.02
    # sim_informed REQUIRES a SIM death prior (lambda_D>0). When none is available
    # (e.g. a real self-declared race axis with no Bridge_R prior -> admin-race deaths
    # can't be placed on it, MSD §2.8.6), degrade to the independent reconstruction
    # rather than fail: the denominator is still the disaggregated strata + closure,
    # just without the death-flow feedback term.
    if mode == "sim_informed_denominator" and sim_deaths is None:
        warnings.append("sim_informed_downgraded_to_independent_no_death_prior")
        mode = "independent_denominator"
    if mode == "sim_informed_denominator":
        feedback_warning = True
        reconstruction_uncertainty = 0.05
        warnings.append("sim_informed_population_feedback_risk")
        rates: list[float | None] = []
        for deaths, anchor in zip(sim_deaths, anchors, strict=True):
            if deaths is None or anchor is None or anchor <= 0:
                rates.append(None)
            else:
                rates.append(float(deaths) / float(anchor))
        death_rates = tuple(rates)

    problem = PopulationTensorProblem(
        shape=shape,
        anchors=tuple(anchors),
        hard_anchor_mask=(False,) * n_cells,
        mode=mode,  # type: ignore[arg-type]
        births=births,
        death_rates=death_rates,
        sim_deaths=sim_deaths,
        race_composition_prior=race_composition_prior,
        closure_totals=tuple(closure),
        migration_locality_totals=migration_locality_totals,
        migration_bounds=tuple(migration_bounds),
        initial_population=tuple(float(value or 0.0) for value in anchors),
        weights=PopulationObjectiveWeights(
            anchor=10.0,
            aging=1.0 if shape[1] > 1 and shape[2] > 1 else 0.0,
            birth=1.0 if births is not None and shape[1] > 1 else 0.0,
            death=1.0 if mode == "sim_informed_denominator" and sim_deaths is not None else 0.0,
            migration=0.1 if shape[1] >= 3 else 0.0,
            migration_total=0.5 if migration_locality_totals is not None else 0.0,
            race=0.1 if race_composition_prior is not None else 0.0,
            age_smooth=0.05 if shape[2] >= 3 else 0.0,
        ),
    )
    solver = select_population_solver(mode=mode, solver_id=solver_id, n_cells=problem.n_cells)
    optimized = solve_population_tensor_problem(
        problem,
        solver_id=solver.solver_id,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    if not optimized.telemetry.converged:
        warnings.append("population_tensor_solver_nonconvergence")
        reconstruction_uncertainty = max(reconstruction_uncertainty, 0.1)

    rows: list[dict[str, Any]] = []
    for locality in localities:
        for period in periods:
            for age_group in age_groups:
                for sex in sexes:
                    for race in races:
                        idx = _cell_index(
                            locality=locality,
                            period=period,
                            age_group=age_group,
                            sex=sex,
                            race=race,
                            locality_index=locality_index,
                            period_index=period_index,
                            age_index=age_index,
                            sex_index=sex_index,
                            race_index=race_index,
                            shape=shape,
                        )
                        rows.append({
                            "year": int(period),
                            "municipality_cod6": locality,
                            "age_group": age_group,
                            "sex": sex,
                            "race": race,
                            "value": float(optimized.population[idx]),
                            "migration": float(optimized.migration[idx]),
                            "anchor_value": anchors[idx],
                            "sim_deaths": None if sim_deaths is None else sim_deaths[idx],
                            "population_tensor_mode": mode,
                            "solver_id": solver.solver_id,
                            "solver_backend": solver.backend,
                            "sparse_jacobian": solver.sparse_jacobian,
                            "denominator_feedback_warning": feedback_warning,
                        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out_path)

    diagnostics = population_tensor_diagnostics(
        request=PopulationTensorRequest(
            mode=mode,  # type: ignore[arg-type]
            solver_id=solver.solver_id,
            solver_backend=solver.backend,
            locality_ids=localities,
            periods=periods,
            strata=tuple(
                f"age_group={age_group}|sex={sex}|race={race}"
                for age_group in age_groups
                for sex in sexes
                for race in races
            ),
        ),
        solver=solver,
        reconstruction_uncertainty=reconstruction_uncertainty,
        denominator_feedback_warning=feedback_warning,
        telemetry=optimized.telemetry,
        warnings=warnings,
    )
    payload = {
        "mode": mode,
        "solver_id": solver.solver_id,
        "shape": shape,
        "population_strata_hash": sha256_file(strata_path),
        "total_anchor_hash": sha256_file(totals_path),
        "sim_events_hash": sha256_file(Path(sim_events_path)) if sim_events_path else None,
        "sinasc_events_hash": sha256_file(Path(sinasc_events_path)) if sinasc_events_path else None,
        "civil_registry_births_hash": sha256_file(Path(civil_registry_births_path)) if civil_registry_births_path else None,
        "civil_registry_deaths_hash": sha256_file(Path(civil_registry_deaths_path)) if civil_registry_deaths_path else None,
        "race_bridge_prior_hash": race_bridge_prior.prior_hash if race_bridge_prior is not None else None,
        "telemetry": optimized.telemetry.as_manifest(),
    }
    result = PopulationTensorResult(
        tensor_id=f"population_tensor_solver_{content_hash(payload)[:24]}",
        mode=mode,  # type: ignore[arg-type]
        solver_id=solver.solver_id,
        solver_backend=solver.backend,
        sparse_jacobian=solver.sparse_jacobian,
        value=float(sum(optimized.population)),
        unit="persons",
        locality_id="panel",
        period="multi" if len(periods) != 1 else periods[0],
        source_anchor_field_id="SIDRA_9606_population_strata",
        source_table_id="9606",
        source_variable_id="93",
        source_request_hash=sha256_file(strata_path),
        source_metadata_hash=sha256_file(totals_path),
        reconstruction_uncertainty=reconstruction_uncertainty,
        denominator_feedback_warning=feedback_warning,
        state="fragile" if warnings else "verified",
        warnings=tuple(warnings),
        diagnostics=diagnostics,
        tensor_shape=shape,
        tensor_values=optimized.population,
        migration_values=optimized.migration,
    )
    migration_flows_path = migration_affinity_path = None
    migration_flow_manifests: tuple[dict[str, Any], ...] = ()
    if reconstruct_migration:
        migration_flows_path, migration_affinity_path, migration_flow_manifests = _reconstruct_and_persist_migration_flows(
            localities=localities,
            periods=periods,
            closure=closure,
            migration_locality_totals=migration_locality_totals,
            shape=shape,
            out_path=out_path,
            contiguity_graph_id=contiguity_graph_id,
            max_hops=migration_max_hops,
        )
    return PopulationTensorBuild(
        result=result,
        output_path=str(out_path),
        locality_ids=localities,
        periods=periods,
        age_groups=age_groups,
        sexes=sexes,
        races=races,
        migration_flows_path=migration_flows_path,
        migration_affinity_path=migration_affinity_path,
        migration_flow_manifests=migration_flow_manifests,
    )


def solve_population_tensor_from_sidra_anchor(
    *,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
    solver_id: str | None = None,
    dense_localities: int | None = None,
    dense_periods: int | None = None,
    dense_strata: int = 1,
) -> PopulationTensorResult:
    """Build a single-cell population tensor from one SIDRA 9606 total anchor fact.

    The single-locality counterpart to :func:`solve_population_tensor_from_sidra_strata`
    (which builds a full ``(locality, time, age, sex, race)`` tensor from disaggregated
    strata) — this is the degenerate case for a single official anchor value.
    """
    sidra_facts_path = Path(sidra_facts_path)
    anchor = load_sidra_population_total_anchor(sidra_facts_path)
    solver = select_population_solver(mode=mode, solver_id=solver_id)

    locality_count = dense_localities if dense_localities is not None else 1
    period_count = dense_periods if dense_periods is not None else 1
    assert_dense_population_tensor_allowed(
        localities=locality_count,
        periods=period_count,
        strata=dense_strata,
        threshold=solver.max_cells,
    )

    request = PopulationTensorRequest(
        mode=mode,  # type: ignore[arg-type]
        solver_id=solver.solver_id,
        solver_backend=solver.backend,
        locality_ids=(anchor.locality_id,),
        periods=(anchor.period,),
        strata=("total",),
        require_sparse=True,
    )

    warnings: list[str] = []
    reconstruction_uncertainty = 0.0
    denominator_feedback_warning = False
    state = "verified"

    if mode == "sim_informed_denominator":
        warnings.extend(("sim_informed_population_feedback_risk", "sim_death_prior_missing_from_sidra_only_request"))
        reconstruction_uncertainty = 0.05
        denominator_feedback_warning = True
        state = "fragile"
    elif mode != "independent_denominator":
        raise ValueError(f"Unsupported population tensor mode: {mode}")

    problem = PopulationTensorProblem(
        shape=(1, 1, 1, 1, 1),
        anchors=(anchor.value,),
        hard_anchor_mask=(False,),
        mode=mode,  # type: ignore[arg-type]
        closure_totals=(anchor.value,),
        migration_bounds=(anchor.value,),
        weights=PopulationObjectiveWeights(death=1.0 if mode == "sim_informed_denominator" else 0.0),
    )
    optimized = solve_population_tensor_problem(problem, solver_id=solver.solver_id)
    if not optimized.telemetry.converged:
        warnings.append("population_tensor_solver_nonconvergence")
        reconstruction_uncertainty = max(reconstruction_uncertainty, 0.1)
        state = "fragile"

    diagnostics = population_tensor_diagnostics(
        request=request,
        solver=solver,
        reconstruction_uncertainty=reconstruction_uncertainty,
        denominator_feedback_warning=denominator_feedback_warning,
        telemetry=optimized.telemetry,
        warnings=warnings,
    )

    payload = {
        "mode": mode,
        "solver_id": solver.solver_id,
        "sidra_facts_path": str(sidra_facts_path),
        "anchor_field_id": anchor.field_id,
        "locality_id": anchor.locality_id,
        "period": anchor.period,
        "value": optimized.population[0],
        "metadata_hash": anchor.metadata_hash,
        "solver_telemetry": optimized.telemetry.as_manifest(),
    }

    return PopulationTensorResult(
        tensor_id=content_hash(payload),
        mode=mode,  # type: ignore[arg-type]
        solver_id=solver.solver_id,
        solver_backend=solver.backend,
        sparse_jacobian=solver.sparse_jacobian,
        value=optimized.population[0],
        unit=anchor.unit,
        locality_id=anchor.locality_id,
        period=anchor.period,
        source_anchor_field_id=anchor.field_id,
        source_table_id=anchor.table_id,
        source_variable_id=anchor.variable_id,
        source_request_hash=anchor.request_hash,
        source_metadata_hash=anchor.metadata_hash,
        reconstruction_uncertainty=reconstruction_uncertainty,
        denominator_feedback_warning=denominator_feedback_warning,
        state=state,
        warnings=tuple(warnings),
        diagnostics=diagnostics,
        tensor_shape=problem.shape,
        tensor_values=optimized.population,
        migration_values=optimized.migration,
    )


__all__ = [
    "PopulationTensorBuild",
    "solve_population_tensor_from_sidra_strata",
    "solve_population_tensor_from_sidra_anchor",
]
