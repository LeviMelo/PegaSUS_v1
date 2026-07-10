"""Population tensor solver orchestration (MSD §2.8).

This module turns real disaggregated SIDRA 9606 facts into a
``PopulationTensorProblem`` over ``(municipality, year, age_group, sex, race)`` and
runs the existing analytic solver. It is intentionally projection-aware: only source
categories that can be mapped through ``demographic/demographic_axis_maps.yaml`` become strata.
Unmapped or unknown categories are excluded rather than silently relabelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.measurement.race import RaceBridgePrior, bridge_admin_race_group_counts, load_race_bridge_prior
from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, age_group_for_years, age_group_sort_key, map_category
from pegasus.registries.population import assert_dense_population_tensor_allowed, select_population_solver
from pegasus.denominators.reconstruction.diagnostics import population_tensor_diagnostics
from pegasus.denominators.reconstruction.schema import (
    PopulationObjectiveWeights,
    PopulationSolverTelemetry,
    PopulationTensorProblem,
    PopulationTensorRequest,
    PopulationTensorResult,
)
from pegasus.denominators.reconstruction.solvers import solve_population_tensor_problem
from pegasus.denominators.population.anchor import (
    geometric_interpolate_closure,
    load_combined_population_totals_frame,
    load_sidra_population_total_anchor,
)

from pegasus.denominators.population.build.strata import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.strata import (
    AXES,
    _canonical_stratum,
    _census_2000_records_from_facts,
    _read_population_strata,
)
from pegasus.denominators.population.build.indexing import *  # noqa: F401,F403 (intra-package base layer)
from pegasus.denominators.population.build.layer1 import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.layer1 import interpolate_census_composition
from pegasus.denominators.population.build.priors import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.priors import (
    _census_race_composition_prior,
    _sim_death_priors,
    _sinasc_birth_priors,
)
from pegasus.denominators.population.build.closure import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.closure import (
    _datasus_event_totals,
    _migration_residual_totals,
    _reanchor_closure_single_vintage,
    _sidra_vital_totals,
)
from pegasus.denominators.population.build.flows import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.flows import _reconstruct_and_persist_migration_flows
from pegasus.denominators.population.build.projection_envelope import *  # noqa: F401,F403 (intra-package layer)
from pegasus.denominators.population.build.projection_envelope import _classify_projection_years


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
# cell through pegasus.measurement.race, never a direct category crosswalk.
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
    # §V: non-None when migration-flow reconstruction was requested but could not run (e.g. the
    # national dense-pair refusal) — recorded so the absent migration-affinity field is never silent.
    migration_flow_skip_reason: str | None = None
    # FAL-POP-PROJ/VER: the census-anchored range [earliest, latest] and the max forward/backward
    # projection horizon (years beyond that range). Feeds the versioned-asset manifest (§VI.2) so a
    # consumer knows which years are enumerated/interpolated vs projected, and to what horizon.
    anchored_range: tuple[int | None, int | None] = (None, None)
    max_projection_horizon: int = 0
    projected_periods: tuple[str, ...] = ()

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
            "migration_flow_skip_reason": self.migration_flow_skip_reason,
            "anchored_range": list(self.anchored_range),
            "max_projection_horizon": self.max_projection_horizon,
            "projected_periods": list(self.projected_periods),
        }


def _per_cell_migration_bounds(
    closure: list[float | None], anchors: np.ndarray, shape: tuple[int, int, int, int, int]
) -> np.ndarray:
    """Per-cell migration bound (MSD §2.8.1): ``0.25*closure_total/strata`` where the year's closure
    total exists (every year), else ``0.25*anchor``, floored at 1.0. Locality-separable — a cell's bound
    depends only on its own (locality,period) closure and its own anchor — so a block's bound is exactly
    the slice of the whole-tensor bound."""
    strata_per_st = shape[2] * shape[3] * shape[4]
    closure_np = np.asarray([c if c is not None else np.nan for c in closure], dtype=np.float64)
    closure_per_cell = np.repeat(closure_np, strata_per_st)
    use_closure = np.isfinite(closure_per_cell) & (closure_per_cell > 0)
    bounds = np.where(
        use_closure,
        0.25 * closure_per_cell / strata_per_st,
        np.nan_to_num(anchors, nan=0.0) * 0.25,
    )
    np.maximum(bounds, 1.0, out=bounds)
    return bounds


def _solve_locality_blocked(
    *,
    localities: tuple[str, ...],
    records_df: pl.DataFrame,
    closure: list[float | None],
    anchors: np.ndarray,
    sim_deaths: Any,
    death_rates: np.ndarray | None,
    births: Any,
    migration_locality_totals: tuple[float | None, ...] | None,
    period_index: dict[str, int],
    age_index: dict[str, int],
    sex_index: dict[str, int],
    race_index: dict[str, int],
    shape: tuple[int, int, int, int, int],
    mode: str,
    weights: PopulationObjectiveWeights,
    informative: bool,
    max_iterations: int,
    tolerance: float,
):
    """Build AND solve the population tensor one locality-block at a time so PEAK memory is O(block),
    not O(national) — §V.1(b) extended upstream to input construction (the measured national RAM cliff:
    ~6 whole 1.08 GB arrays coexisting). The SIDRA objective is locality-separable (``_locality_separable``:
    no cross-locality migration enclosure), so each block's sub-problem equals ``_slice_localities(
    full_problem, s0, s1)`` EXACTLY — the per-block prior_mean / race composition / migration bounds are
    the corresponding slices of the whole-tensor arrays, and the per-block solve/project is byte-identical
    to solving the sliced sub-problem. anchors + the (already-materialized, mostly-None) flow priors stay
    whole and are sliced; only the three dominant O(n_cells) priors are built per block and freed.
    Block sizing + per-block solver reselection mirror ``solve_population_tensor_blocked`` so telemetry
    matches too. Returns a ``PopulationOptimizationResult`` with the assembled whole-tensor pop/migration."""
    from pegasus.denominators.reconstruction.solvers import (
        _BLOCK_TARGET_CELLS,
        _GPU_MAX_SAFE_BLOCKS,
        solve_population_tensor_problem,
    )
    from pegasus.denominators.reconstruction.projected_gradient import (
        PopulationOptimizationResult,
        _fast_projection_supported,
        _np_project_population,
        _project_population,
    )

    s_count, t_count, a_count, x_count, r_count = shape
    inner = a_count * x_count * r_count
    birth_inner = x_count * r_count
    n_cells = s_count * t_count * inner
    per_locality = max(1, n_cells // max(1, s_count))
    block_localities = max(1, _BLOCK_TARGET_CELLS // per_locality)
    # POP-02 GPU crash guard: the per-block GPU solve is a validated ~16x win at study/state scale, but
    # the full-national many-block loop (~68 blocks) intermittently hard-segfaults from a native
    # torch+polars(rayon) transition race. Route the largest builds to the crash-free CPU path; keep the
    # GPU where it is tested-safe (<=~single UF, ~11 blocks). `_GPU_MAX_SAFE_BLOCKS` is that boundary;
    # the robust fix (subprocess isolation / race elimination) would re-enable national GPU.
    n_blocks_total = (s_count + block_localities - 1) // block_localities
    prefer_gpu = n_blocks_total <= _GPU_MAX_SAFE_BLOCKS

    sim_full = None if sim_deaths is None else np.asarray(sim_deaths, dtype=np.float64)
    births_full = None if births is None else np.asarray(births, dtype=np.float64)
    dr_full = None if death_rates is None else np.asarray(death_rates, dtype=np.float64)

    full_pop = np.zeros(n_cells, dtype=np.float64)
    full_mig = np.zeros(n_cells, dtype=np.float64)
    converged_all, iters_max, init_obj, final_obj, worst_grad, n_blocks = True, 0, 0.0, 0.0, 0.0, 0
    last_telemetry = None
    for s0 in range(0, s_count, block_localities):
        s1 = min(s0 + block_localities, s_count)
        block_locs = localities[s0:s1]
        bshape = (s1 - s0, t_count, a_count, x_count, r_count)
        bloc_index = {loc: i for i, loc in enumerate(block_locs)}
        brecords = records_df.filter(pl.col("municipality_cod6").is_in(list(block_locs)))
        bclosure = closure[s0 * t_count:s1 * t_count]
        c0, c1 = s0 * t_count * inner, s1 * t_count * inner
        st0, st1 = s0 * t_count, s1 * t_count
        bh0, bh1 = s0 * t_count * birth_inner, s1 * t_count * birth_inner
        b_anchors = np.asarray(anchors[c0:c1], dtype=np.float64)
        b_prior_mean = interpolate_census_composition(
            records_df=brecords, closure=bclosure, locality_index=bloc_index, period_index=period_index,
            age_index=age_index, sex_index=sex_index, race_index=race_index, shape=bshape,
        )
        b_race_prior = _census_race_composition_prior(
            records_df=brecords, locality_index=bloc_index, period_index=period_index,
            age_index=age_index, sex_index=sex_index, race_index=race_index, shape=bshape,
        )
        b_problem = PopulationTensorProblem(
            shape=bshape,
            anchors=b_anchors,
            hard_anchor_mask=None,
            mode=mode,  # type: ignore[arg-type]
            births=None if births_full is None else births_full[bh0:bh1],
            death_rates=None if dr_full is None else dr_full[c0:c1],
            sim_deaths=None if sim_full is None else sim_full[c0:c1],
            race_composition_prior=b_race_prior,
            closure_totals=np.asarray([c if c is not None else np.nan for c in bclosure], dtype=np.float64),
            migration_locality_totals=None if migration_locality_totals is None else migration_locality_totals[st0:st1],
            migration_bounds=_per_cell_migration_bounds(bclosure, b_anchors, bshape),
            initial_population=np.asarray(b_prior_mean, dtype=np.float64),
            weights=weights,
        )
        if informative:
            opt = solve_population_tensor_problem(
                b_problem, solver_id=None, max_iterations=max_iterations, tolerance=tolerance,
                prefer_gpu=prefer_gpu,
            )
            last_telemetry = opt.telemetry
            converged_all = converged_all and opt.telemetry.converged
            iters_max = max(iters_max, opt.telemetry.iterations)
            init_obj += opt.telemetry.initial_objective
            final_obj += opt.telemetry.final_objective
            worst_grad = max(worst_grad, opt.telemetry.projected_gradient_norm)
            full_pop[c0:c1] = np.asarray(opt.population, dtype=np.float64)
            full_mig[c0:c1] = np.asarray(opt.migration, dtype=np.float64)
        else:
            prior_np = np.asarray(b_prior_mean, dtype=np.float64)
            full_pop[c0:c1] = (
                _np_project_population(b_problem, prior_np)
                if _fast_projection_supported(b_problem)
                else np.asarray(_project_population(b_problem, prior_np.tolist()), dtype=np.float64)
            )  # full_mig stays zero for the data-poor closed-form block
        n_blocks += 1

    if not informative:
        telemetry = PopulationSolverTelemetry(
            converged=True, iterations=0, initial_objective=0.0, final_objective=0.0,
            projected_gradient_norm=0.0, relative_objective_change=0.0, step_size=0.0, objective_terms={},
        )
    elif n_blocks == 1 and last_telemetry is not None:
        # Single whole-panel block (small scope): preserve the solver's own per-term objective_terms
        # (anchor/aging/birth/death/race/migration/age_smooth) EXACTLY as solve_population_tensor_blocked's
        # whole-problem fallback did -- callers read these terms to confirm which losses fired.
        telemetry = last_telemetry
    else:
        # Multi-block national solve: the per-term breakdown isn't summable across blocks, so report the
        # block structure (identical to the old solve_population_tensor_blocked aggregation).
        telemetry = PopulationSolverTelemetry(
            converged=converged_all, iterations=iters_max, initial_objective=init_obj,
            final_objective=final_obj, projected_gradient_norm=worst_grad,
            relative_objective_change=0.0, step_size=0.0,
            objective_terms={"blocked_localities": float(block_localities), "n_blocks": float(n_blocks)},
        )
    return PopulationOptimizationResult(full_pop, full_mig, telemetry)


def solve_population_tensor_from_sidra_strata(
    *,
    population_strata_path: str | Path,
    total_anchor_path: str | Path,
    output_path: str | Path,
    census_2000_strata_path: str | Path | None = None,
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
    max_iterations: int = 12,
    tolerance: float = 1e-5,
) -> PopulationTensorBuild:
    # Iteration budget note (MSD §2.8.10): with a single census in the window and free
    # migration, the intercensal (a,x,r) structure is underdetermined -- the aging/race/
    # smoothness losses cannot reduce their residual below a floor, so the solver never hits
    # a tight tolerance and would otherwise run to a huge cap. The demographically-correct
    # reconstruction is the census-proportion warm start (see projected_gradient._initial_
    # population); a small number of refinement steps is sufficient and well-conditioned runs
    # (informative flows / multiple censuses) still converge and early-stop before the cap.
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
    del strata  # the 11.25M-row source frame is fully consumed into `records`; free it (~1.5 GB national)
    # before the 2000-census disaggregation doubles the record set (RAM: §V.1/§VIII).

    # FAL-POP #4 (§II.4): fold in the 2000 census (SIDRA 2093) as a third anchor, disaggregated from
    # its coarse age brackets to single year using the 2010 shape. Appended to the census record set
    # so the solver cohort-ages across 2000/2010/2022. Absent 2093 → no 2000 anchor (build uses 2010/2022).
    census_2000_recon: dict[str, float] = {}
    if census_2000_strata_path is not None:
        records_2000, census_2000_recon = _census_2000_records_from_facts(census_2000_strata_path, records)
        for record in records_2000:
            records.append(record)
            for axis in AXES:
                categories_by_axis[axis].add(record[axis])

    # FAL-POP-AMC (§II.4): carve municipalities installed AFTER a census out of their parents, so a
    # census-year total stays the enumerated total instead of double-counting the child's people (which
    # were counted inside the parents). The child→parent map is authoritative (IBGE territorial
    # evolution), never inferred. Mass-preserving: parents lose X, child gains X.
    from pegasus.denominators.population.census_2000 import (
        carve_pre_census_children,
        load_amc_crosswalk,
        load_municipality_genealogy_overrides,
    )

    amc_stats = carve_pre_census_children(
        records, load_amc_crosswalk(), load_municipality_genealogy_overrides()
    )
    for record in records:  # the carve may introduce a child's cells at a new census period
        for axis in AXES:
            categories_by_axis[axis].add(record[axis])

    # §V.1 / §VIII: the record set is now complete (9606 + 2000 census + AMC carve). Collapse the
    # ~12M-row list[dict] (~6 GB at national scale) into a columnar frame and free the list, so the
    # downstream numeric consumers (anchor scatter, census-count arrays, race prior) run vectorized and
    # never coexist with the O(n_cells) tensors as fat Python dicts -- the measured national RAM cliff
    # (peak ~18 GB -> thrash). Columns are keyed by NAME, so the parse-order and census_2000-order
    # record dicts align identically.
    records_df = pl.DataFrame(
        records,
        schema={"municipality_cod6": pl.Utf8, "period": pl.Utf8, "value": pl.Float64,
                "age_group": pl.Utf8, "sex": pl.Utf8, "race": pl.Utf8},
    )
    del records

    # `totals` stitches the census-year (9606) and intercensal (6579) population
    # totals into one (municipality, year) closure panel (MSD §2.8.10); its period
    # coverage is a superset of the strata's (strata/disaggregation only exists for
    # census years) -- the tensor's time axis must span BOTH so intercensal years
    # get a real closure anchor instead of silently having none.
    totals = load_combined_population_totals_frame(totals_path)
    localities = tuple(sorted(set(records_df["municipality_cod6"].to_list()) | {str(row) for row in totals["municipality_cod6"].to_list()}))
    periods = tuple(sorted(set(records_df["period"].to_list()) | {str(int(row)) for row in totals["year"].to_list()}))
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
    inner_cells = shape[2] * shape[3] * shape[4]
    # Vectorized numpy anchor construction (§V.1): the flat cell index is computed for every record at
    # once and scatter-added, so a national ~1.3e8-cell anchor field is a single float64 array (NaN =
    # absent) instead of a Python list of ~1.3e8 objects, and the fill is one np.add.at not a per-record
    # loop. Byte-identical to the old accumulate-per-record.
    _aidx = records_df.select(
        pl.col("municipality_cod6").cast(pl.Utf8).replace_strict(locality_index, default=-1, return_dtype=pl.Int64).alias("li"),
        pl.col("period").cast(pl.Utf8).replace_strict(period_index, default=-1, return_dtype=pl.Int64).alias("pi"),
        pl.col("age_group").cast(pl.Utf8).replace_strict(age_index, default=-1, return_dtype=pl.Int64).alias("ai"),
        pl.col("sex").cast(pl.Utf8).replace_strict(sex_index, default=-1, return_dtype=pl.Int64).alias("xi"),
        pl.col("race").cast(pl.Utf8).replace_strict(race_index, default=-1, return_dtype=pl.Int64).alias("ri"),
        pl.col("value").cast(pl.Float64).alias("rv"),
    )
    li = _aidx["li"].to_numpy(); pi = _aidx["pi"].to_numpy(); ai = _aidx["ai"].to_numpy()
    xi = _aidx["xi"].to_numpy(); ri = _aidx["ri"].to_numpy(); rvals = _aidx["rv"].to_numpy()
    valid = (li >= 0) & (pi >= 0) & (ai >= 0) & (xi >= 0) & (ri >= 0)
    flat_idx = ((li * shape[1] + pi) * inner_cells) + (ai * (shape[3] * shape[4]) + xi * shape[4] + ri)
    anchors = np.zeros(n_cells, dtype=np.float64)
    np.add.at(anchors, flat_idx[valid], rvals[valid])
    touched = np.zeros(n_cells, dtype=bool)
    touched[flat_idx[valid]] = True
    anchors[~touched] = np.nan  # NaN = no anchor (was None); accumulated sum otherwise

    closure: list[float | None] = [None] * (shape[0] * shape[1])
    for row in totals.iter_rows(named=True):
        locality = str(row["municipality_cod6"])
        period = str(row["year"])
        if locality in locality_index and period in period_index:
            closure[locality_index[locality] * shape[1] + period_index[period]] = float(row["value"])
    # Strata-sum fallback for closure cells no totals table covers: sum the year's anchors (NaN-aware).
    grouped = anchors.reshape(shape[0] * shape[1], inner_cells)
    strata_sum = np.nansum(grouped, axis=1)
    has_strata = ~np.isnan(grouped).all(axis=1)
    for idx in range(len(closure)):
        if closure[idx] is None and bool(has_strata[idx]):
            closure[idx] = float(strata_sum[idx])

    # FAL-POP-SV (§II.4 anti-discontinuity contract): re-anchor the intercensal closure to the
    # CENSUS vintage. The panel above still carries SIDRA-6579 *projection*-vintage totals for
    # intercensal years -- 6579's pre-census projection was revised down ~10M by the 2022 census,
    # so anchoring 2021 to 6579 (~213M) and 2022 to the census (~203M) injects a spurious ~5%
    # denominator jump that reads as a fake trend. Replace every non-census year with geometric
    # interpolation between the bounding census enumerations (per municipality), putting the whole
    # series on one vintage; 6579/EstimaPOP survives only as a validation cross-check (below), never
    # the anchor. Census years (2000/2010/2022 -- the years the strata records cover) are untouched.
    # Municipalities with <2 census anchors (created after 2000, boundary churn) keep their prior
    # closure and are counted for the caller's telemetry.
    census_years = frozenset(records_df["period"].to_list())
    single_vintage_stats = _reanchor_closure_single_vintage(
        closure=closure,
        localities=localities,
        periods=periods,
        census_years=census_years,
        shape=shape,
        period_index=period_index,
    )

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
    # The race composition prior is built PER BLOCK in _solve_locality_blocked (§V.1(b) O(block) RAM).
    # Its presence -- which sets the race loss weight below -- is exactly the condition on which
    # _census_race_composition_prior returns non-None: a real (non-TOTAL) race axis with >=2 categories.
    has_race_prior = any(v != TOTAL for v in race_index) and len(race_index) >= 2
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
    # Per-cell migration bound (MSD §2.8.1, closure-based) is built PER BLOCK in _solve_locality_blocked
    # via _per_cell_migration_bounds -- it was a whole-tensor ~1.08 GB array (§V.1(b) O(block) RAM).
    death_rates: tuple[float | None, ...] | None = None
    warnings: list[str] = [*death_warnings, *birth_warnings, *migration_warnings]
    # FAL-POP-RECON telemetry (§II.5): the undeclared-race mass reallocated into the declared races so
    # the 2000 anchor sums to the enumerated total (census-exact), and any all-undeclared cells that
    # fell back to a broader composition.
    if census_2000_recon.get("undeclared_reallocated", 0.0) > 0:
        warnings.append(
            "census_2000_undeclared_race_reconciled"
            f"::reallocated={census_2000_recon['undeclared_reallocated']:.0f}"
            f"::fallback_cells={int(census_2000_recon.get('fallback_cells', 0))}"
        )
    if census_2000_recon.get("undeclared_dropped_no_declared", 0.0) > 0:
        warnings.append(
            "census_2000_undeclared_race_unreconciled_no_declared"
            f"::mass={census_2000_recon['undeclared_dropped_no_declared']:.0f}"
        )
    # FAL-POP-AMC telemetry: post-census municipalities carved out of their parents (mass-preserving).
    if amc_stats.get("amc_children_carved", 0) > 0:
        warnings.append(
            "census_boundary_amc_carved"
            f"::children={int(amc_stats['amc_children_carved'])}"
            f"::population={amc_stats['amc_carved_population']:.0f}"
        )
    # FAL-POP-SV telemetry: record that the intercensal closure is on the census (not 6579) vintage,
    # and flag municipalities that could not be re-anchored (kept on their prior 6579 closure).
    if single_vintage_stats["reanchored_municipalities"] > 0:
        warnings.append(
            "closure_single_vintage_census_anchored"
            f"::reanchored={single_vintage_stats['reanchored_municipalities']}"
            f"::single_anchor_kept={single_vintage_stats['kept_single_anchor_municipalities']}"
        )
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
        # Vectorized rate = deaths / anchor where both present and anchor > 0, else NaN (was None).
        sim_np = np.asarray([np.nan if d is None else float(d) for d in sim_deaths], dtype=np.float64)
        rate_mask = ~np.isnan(sim_np) & ~np.isnan(anchors) & (anchors > 0)
        death_rates_arr = np.full(n_cells, np.nan, dtype=np.float64)
        death_rates_arr[rate_mask] = sim_np[rate_mask] / anchors[rate_mask]
        death_rates = death_rates_arr

    # Layer 1 (MSD §2.8.10): the closed-form prior-mean tensor (census composition interpolated across
    # years, scaled to each closure total -- data-poor reconstruction / solver warm start) is built PER
    # BLOCK in _solve_locality_blocked; records_df is consumed there and freed after the solve.
    census_year_count = len(census_years)

    weights = PopulationObjectiveWeights(
        anchor=10.0,
        aging=1.0 if shape[1] > 1 and shape[2] > 1 else 0.0,
        birth=1.0 if births is not None and shape[1] > 1 else 0.0,
        death=1.0 if mode == "sim_informed_denominator" and sim_deaths is not None else 0.0,
        migration=0.1 if shape[1] >= 3 else 0.0,
        migration_total=0.5 if migration_locality_totals is not None else 0.0,
        race=0.1 if has_race_prior else 0.0,
        age_smooth=0.05 if shape[2] >= 3 else 0.0,
    )
    # Pass numpy arrays (not Python tuples): PopulationTensorProblem stores them as-is (§V.1),
    # so the O(n_cells) inputs never materialize as ~32 B/element Python float tuples. hard_anchor_mask
    # is left None (no hard anchors in the SIDRA build) rather than an all-False n_cells vector.
    solver = select_population_solver(mode=mode, solver_id=solver_id, n_cells=n_cells)
    # Layer 2: refine the prior mean ONLY when a flow term carries data (multiple censuses to
    # cohort-age between, a SIM death prior, births, or an observed net-migration residual).
    # Absent all of them the (a,x,r) structure is underdetermined and its optimum IS the prior
    # mean -- so skip the (flat, slow) optimization and return the closed-form layer directly.
    informative = (
        census_year_count >= 2
        or weights.death > 0.0
        or weights.birth > 0.0
        or weights.migration_total > 0.0
    )
    # §V.1(b): build AND solve in locality-blocks so PEAK memory is O(block), not O(national). The
    # dominant O(n_cells) priors (prior_mean / race composition / migration bounds) are constructed per
    # block and freed -- never coexisting whole -- which was the measured national RAM cliff. Exact for
    # the locality-separable SIDRA objective (byte-identical to the whole-tensor build; see
    # _solve_locality_blocked). Solver metadata (``solver``) is still selected on the FULL cell count.
    optimized = _solve_locality_blocked(
        localities=localities, records_df=records_df, closure=closure, anchors=anchors,
        sim_deaths=sim_deaths, death_rates=death_rates, births=births,
        migration_locality_totals=migration_locality_totals, period_index=period_index,
        age_index=age_index, sex_index=sex_index, race_index=race_index, shape=shape,
        mode=mode, weights=weights, informative=informative,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    import gc as _gc
    del records_df  # consumed per-block above; free before the emit
    _gc.collect()
    if informative:
        if not optimized.telemetry.converged:
            warnings.append("population_tensor_solver_nonconvergence")
            reconstruction_uncertainty = max(reconstruction_uncertainty, 0.1)
    else:
        warnings.append("population_reconstruction_closed_form_no_informative_flows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Streamed tensor emission (§V.1 M6): the flat cell order (locality, period, age, sex, race) is
    # row-major, so each locality is a contiguous ``t*inner``-cell slab. Emit LOCALITY-BLOCKS through one
    # ParquetWriter; each block's label columns are O(block), so a national ~1.3e8-cell tensor never
    # materializes the full ~8 GB label frame. Byte-identical to a monolithic emit (same row order).
    import pyarrow.parquet as pq

    s_count, t_count, a_count, x_count, r_count = shape
    inner = a_count * x_count * r_count
    pop = np.asarray(optimized.population, dtype=np.float64)
    mig = np.asarray(optimized.migration, dtype=np.float64)
    anch = np.asarray(anchors, dtype=np.float64)
    simd = (
        np.full(pop.shape[0], np.nan) if sim_deaths is None
        else np.asarray([np.nan if d is None else float(d) for d in sim_deaths], dtype=np.float64)
    )
    # FAL-POP-PROJ (§II.4): per-year anchor classification + horizon-growing uncertainty envelope.
    proj_map, anchored_range, max_horizon = _classify_projection_years(
        periods, census_years, reconstruction_uncertainty
    )
    projected_periods = [p for p in periods if proj_map[p][0].startswith("projected")]
    if projected_periods:
        warnings.append(
            "population_tensor_projected_years"
            f"::count={len(projected_periods)}::max_horizon={max_horizon}"
            f"::anchored_range={anchored_range[0]}-{anchored_range[1]}"
        )
    # One-locality (t*inner) label templates -- identical for every locality; only the muni id varies.
    one_year = np.repeat(np.array([int(p) for p in periods], dtype=np.int64), inner)
    one_age = np.tile(np.repeat(np.asarray(age_groups, dtype=object), x_count * r_count), t_count)
    one_sex = np.tile(np.repeat(np.asarray(sexes, dtype=object), r_count), t_count * a_count)
    one_race = np.tile(np.asarray(races, dtype=object), t_count * a_count * x_count)
    one_cls = np.repeat(np.array([proj_map[p][0] for p in periods], dtype=object), inner)
    one_hor = np.repeat(np.array([proj_map[p][1] for p in periods], dtype=np.int64), inner)
    one_state = np.repeat(np.array([proj_map[p][2] for p in periods], dtype=object), inner)
    one_unc = np.repeat(np.array([proj_map[p][3] for p in periods], dtype=np.float64), inner)
    loc_arr = np.asarray(localities, dtype=object)
    slab = t_count * inner
    emit_block = max(1, 2_000_000 // slab)
    out_schema = {
        "year": pl.Int64, "municipality_cod6": pl.Utf8, "age_group": pl.Utf8, "sex": pl.Utf8,
        "race": pl.Utf8, "value": pl.Float64, "migration": pl.Float64,
        "anchor_value": pl.Float64, "sim_deaths": pl.Float64,
        "anchor_class": pl.Utf8, "projection_horizon": pl.Int64,
        "cell_state": pl.Utf8, "cell_uncertainty": pl.Float64,
    }
    writer = None
    for s0 in range(0, s_count, emit_block):
        s1 = min(s0 + emit_block, s_count)
        nloc = s1 - s0
        lo, hi = s0 * slab, s1 * slab
        frame = pl.DataFrame(
            {
                "year": np.tile(one_year, nloc),
                "municipality_cod6": np.repeat(loc_arr[s0:s1], slab),
                "age_group": np.tile(one_age, nloc), "sex": np.tile(one_sex, nloc), "race": np.tile(one_race, nloc),
                "value": pop[lo:hi], "migration": mig[lo:hi], "anchor_value": anch[lo:hi], "sim_deaths": simd[lo:hi],
                "anchor_class": np.tile(one_cls, nloc), "projection_horizon": np.tile(one_hor, nloc),
                "cell_state": np.tile(one_state, nloc), "cell_uncertainty": np.tile(one_unc, nloc),
            },
            schema=out_schema,
        ).with_columns(
            pl.col("anchor_value").fill_nan(None),
            pl.col("sim_deaths").fill_nan(None),
            pl.lit(mode).alias("population_tensor_mode"),
            pl.lit(solver.solver_id).alias("solver_id"),
            pl.lit(solver.backend).alias("solver_backend"),
            pl.lit(bool(solver.sparse_jacobian)).alias("sparse_jacobian"),
            pl.lit(bool(feedback_warning)).alias("denominator_feedback_warning"),
        )
        table = frame.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()

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
        value=float(np.asarray(optimized.population, dtype=np.float64).sum()),
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
    migration_flow_skip_reason: str | None = None
    if reconstruct_migration:
        (
            migration_flows_path,
            migration_affinity_path,
            migration_flow_manifests,
            migration_flow_skip_reason,
        ) = _reconstruct_and_persist_migration_flows(
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
        migration_flow_skip_reason=migration_flow_skip_reason,
        anchored_range=anchored_range,
        max_projection_horizon=max_horizon,
        projected_periods=tuple(projected_periods),
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
    "SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE",
    "SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE",
    "SIDRA_CIVIL_REGISTRY_DEATHS_TABLE",
    "SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE",
]
