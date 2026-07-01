from __future__ import annotations

from pegasus.registries.population import (
    DENSE_NATIONAL_CELL_THRESHOLD,
    assert_dense_population_tensor_allowed,
    select_population_solver,
)
from pegasus.she.reconstruction.projected_gradient import PopulationOptimizationResult, solve_projected_gradient_small
from pegasus.she.reconstruction.schema import PopulationTensorProblem
from pegasus.she.reconstruction.sparse_admm import solve_population_admm, solve_sparse_population
from pegasus.she.reconstruction.state_space import solve_population_state_space_smoother


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


def dense_national_abort_check(*, localities: int, periods: int, strata: int = 1) -> int:
    return assert_dense_population_tensor_allowed(
        localities=localities,
        periods=periods,
        strata=strata,
        threshold=DENSE_NATIONAL_CELL_THRESHOLD,
    )


__all__ = ["solve_population_tensor_problem", "dense_national_abort_check"]
