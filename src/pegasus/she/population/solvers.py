from __future__ import annotations

from pathlib import Path

from pegasus.core.hashing import content_hash
from pegasus.registries.population import (
    DENSE_NATIONAL_CELL_THRESHOLD,
    assert_dense_population_tensor_allowed,
    select_population_solver,
)
from pegasus.she.population.diagnostics import population_tensor_diagnostics
from pegasus.she.population.projected_gradient import PopulationOptimizationResult, solve_projected_gradient_small
from pegasus.she.population.schema import (
    PopulationObjectiveWeights,
    PopulationTensorProblem,
    PopulationTensorRequest,
    PopulationTensorResult,
)
from pegasus.she.population.sidra_anchor import load_sidra_population_total_anchor
from pegasus.she.population.sparse_admm import solve_population_admm, solve_sparse_population
from pegasus.she.population.state_space import solve_population_state_space_smoother


def solve_population_tensor_problem(
    problem: PopulationTensorProblem,
    *,
    solver_id: str | None = None,
    max_iterations: int = 5_000,
    tolerance: float = 1e-5,
) -> PopulationOptimizationResult:
    solver = select_population_solver(mode=problem.mode, solver_id=solver_id, n_cells=problem.n_cells)
    if problem.n_cells > solver.max_cells:
        raise ValueError(f"Population problem cells={problem.n_cells} exceeds solver policy max_cells={solver.max_cells}.")
    if solver.backend.startswith("projected_gradient_small"):
        assert_dense_population_tensor_allowed(
            localities=problem.shape[0], periods=problem.shape[1],
            strata=problem.shape[2] * problem.shape[3] * problem.shape[4], threshold=solver.max_cells,
        )
        return solve_projected_gradient_small(problem, max_iterations=max_iterations, tolerance=tolerance)
    if solver.backend.startswith("sparse_block_coordinate"):
        result, _ = solve_sparse_population(
            problem, solver_id=solver.solver_id, max_iterations=max_iterations, tolerance=tolerance,
        )
        return result
    if solver.backend.startswith("sparse_admm"):
        result, _ = solve_population_admm(
            problem, solver_id=solver.solver_id, max_iterations=max_iterations, tolerance=tolerance,
        )
        return result
    if solver.backend.startswith("state_space_smoother"):
        smoothed = solve_population_state_space_smoother(problem)
        return smoothed.result
    raise ValueError(f"Solver {solver.solver_id} does not implement population optimization.")


def solve_population_tensor_from_sidra_anchor(
    *,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
    solver_id: str | None = None,
    dense_localities: int | None = None,
    dense_periods: int | None = None,
    dense_strata: int = 1,
) -> PopulationTensorResult:
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


def dense_national_abort_check(*, localities: int, periods: int, strata: int = 1) -> int:
    return assert_dense_population_tensor_allowed(
        localities=localities,
        periods=periods,
        strata=strata,
        threshold=DENSE_NATIONAL_CELL_THRESHOLD,
    )
