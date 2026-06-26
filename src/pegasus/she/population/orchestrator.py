"""Population tensor solver orchestration (MSD §2.8).

This module turns real disaggregated SIDRA 9606 facts into a
``PopulationTensorProblem`` over ``(municipality, year, age_group, sex, race)`` and
runs the existing analytic solver. It is intentionally projection-aware: only source
categories that can be mapped through ``demographic_axis_maps.yaml`` become strata.
Unmapped or unknown categories are excluded rather than silently relabelled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, map_category
from pegasus.she.population.diagnostics import population_tensor_diagnostics
from pegasus.she.population.schema import (
    PopulationObjectiveWeights,
    PopulationTensorProblem,
    PopulationTensorRequest,
    PopulationTensorResult,
)
from pegasus.she.population.sidra_anchor import load_sidra_population_totals_frame
from pegasus.she.population.solvers import solve_population_tensor_problem
from pegasus.registries.population import select_population_solver


AXES = ("age_group", "sex", "race")
AXIS_CLASSIFICATIONS = {"sex": "2", "race": "86", "age_group": "287"}


@dataclass(frozen=True)
class PopulationTensorBuild:
    result: PopulationTensorResult
    output_path: str
    locality_ids: tuple[str, ...]
    periods: tuple[str, ...]
    age_groups: tuple[str, ...]
    sexes: tuple[str, ...]
    races: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            **self.result.as_manifest(),
            "output_path": self.output_path,
            "locality_ids": list(self.locality_ids),
            "periods": list(self.periods),
            "age_groups": list(self.age_groups),
            "sexes": list(self.sexes),
            "races": list(self.races),
        }


def _pairs(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    return [tuple(map(str, item)) for item in (parsed or []) if len(item) >= 2]


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


def _sim_death_priors(
    *,
    sim_events_path: str | Path | None,
    locality_index: dict[str, int],
    period_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    shape: tuple[int, int, int, int, int],
) -> tuple[float | None, ...] | None:
    if sim_events_path is None:
        return None
    path = Path(sim_events_path)
    if not path.exists():
        return None
    from pegasus.registries.demographic_axis import source_category_map

    frame = pl.read_parquet(path)
    if "municipality_cod6" not in frame.columns:
        geo_column = next(
            (column for column in ("mun_residence_cod6", "mun_occurrence_cod6", "CODMUNRES", "MUNIC_RES") if column in frame.columns),
            None,
        )
        if geo_column is None:
            return None
        frame = frame.with_columns(pl.col(geo_column).cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("municipality_cod6"))
    if "year" not in frame.columns:
        year_column = next((column for column in ("death_year", "event_year", "ANO") if column in frame.columns), None)
        if year_column is None:
            return None
        frame = frame.with_columns(pl.col(year_column).cast(pl.Int64, strict=False).alias("year"))
    if "municipality_cod6" not in frame.columns or "year" not in frame.columns:
        return None
    sex_map = source_category_map("sex", "SIM-DO")
    if "sex" in frame.columns and any(value != TOTAL for value in sex_index) and sex_map:
        frame = frame.with_columns(
            pl.col("sex").cast(pl.Utf8, strict=False).replace(sex_map).alias("__sex__")
        ).filter(pl.col("__sex__").is_in(list(sex_index)))
    else:
        frame = frame.with_columns(pl.lit(TOTAL).alias("__sex__"))
    grouped = frame.group_by(["municipality_cod6", "year", "__sex__"]).agg(pl.len().cast(pl.Float64).alias("deaths"))
    values: list[float | None] = [None] * (shape[0] * shape[1] * shape[2] * shape[3] * shape[4])
    for row in grouped.iter_rows(named=True):
        locality = str(row["municipality_cod6"])
        period = str(row["year"])
        sex = str(row["__sex__"])
        if locality not in locality_index or period not in period_index or sex not in sex_index:
            continue
        idx = _cell_index(
            locality=locality,
            period=period,
            age_group=TOTAL,
            sex=sex,
            race=TOTAL,
            locality_index=locality_index,
            period_index=period_index,
            age_index=age_index,
            sex_index=sex_index,
            race_index=race_index,
            shape=shape,
        )
        values[idx] = float(row["deaths"])
    return tuple(values)


def solve_population_tensor_from_sidra_strata(
    *,
    population_strata_path: str | Path,
    total_anchor_path: str | Path,
    output_path: str | Path,
    mode: str = "independent_denominator",
    sim_events_path: str | Path | None = None,
    solver_id: str | None = None,
    max_iterations: int = 2_000,
    tolerance: float = 1e-5,
) -> PopulationTensorBuild:
    strata_path = Path(population_strata_path)
    totals_path = Path(total_anchor_path)
    out_path = Path(output_path)
    if mode not in {"independent_denominator", "sim_informed_denominator"}:
        raise ValueError(f"Unsupported population tensor solver mode: {mode}")

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

    localities = tuple(sorted({record["municipality_cod6"] for record in records}))
    periods = tuple(sorted({record["period"] for record in records}))
    age_groups = tuple(sorted(categories_by_axis["age_group"] or {TOTAL}))
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

    totals = load_sidra_population_totals_frame(totals_path)
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

    sim_deaths = _sim_death_priors(
        sim_events_path=sim_events_path,
        locality_index=locality_index,
        period_index=period_index,
        age_index=age_index,
        sex_index=sex_index,
        race_index=race_index,
        shape=shape,
    )
    death_rates: tuple[float | None, ...] | None = None
    warnings: list[str] = []
    feedback_warning = False
    reconstruction_uncertainty = 0.02
    if mode == "sim_informed_denominator":
        feedback_warning = True
        reconstruction_uncertainty = 0.05
        warnings.append("sim_informed_population_feedback_risk")
        if sim_deaths is None:
            warnings.append("sim_death_prior_missing")
        else:
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
        death_rates=death_rates,
        sim_deaths=sim_deaths,
        closure_totals=tuple(closure),
        migration_bounds=tuple(max((value or 0.0) * 0.25, 1.0) for value in anchors),
        initial_population=tuple(float(value or 0.0) for value in anchors),
        weights=PopulationObjectiveWeights(
            anchor=10.0,
            aging=1.0 if shape[1] > 1 and shape[2] > 1 else 0.0,
            birth=0.0,
            death=1.0 if mode == "sim_informed_denominator" and sim_deaths is not None else 0.0,
            migration=0.1 if shape[1] >= 3 else 0.0,
            race=0.0,
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
    return PopulationTensorBuild(
        result=result,
        output_path=str(out_path),
        locality_ids=localities,
        periods=periods,
        age_groups=age_groups,
        sexes=sexes,
        races=races,
    )


__all__ = ["PopulationTensorBuild", "solve_population_tensor_from_sidra_strata"]
