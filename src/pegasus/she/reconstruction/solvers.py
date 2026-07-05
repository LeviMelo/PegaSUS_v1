from __future__ import annotations

import numpy as np

from pegasus.registries.population import (
    DENSE_NATIONAL_CELL_THRESHOLD,
    assert_dense_population_tensor_allowed,
    select_population_solver,
)
from pegasus.she.reconstruction.projected_gradient import PopulationOptimizationResult, solve_projected_gradient_small
from pegasus.she.reconstruction.schema import PopulationSolverTelemetry, PopulationTensorProblem
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


# §V.1(b): target cells per block. The dense sub-solver holds ~15 float64 working arrays, so this
# bounds a block's working set to ~250 MB — small enough to fit even a loaded box, and well under
# the 10M-cell dense-solver ceiling so every block takes the fast dense path.
_BLOCK_TARGET_CELLS = 2_000_000


def _slice_localities(problem: PopulationTensorProblem, s0: int, s1: int) -> PopulationTensorProblem:
    """Extract the sub-problem for localities ``[s0, s1)``. The population objective couples cells
    only *within* a locality (aging/birth/death/race/smoothness are all per-locality; closure and
    the net-migration residual are per locality-year) — so a locality slice is an exact independent
    sub-problem. Flat order is (s,t,a,x,r) with s outermost, so a locality range is contiguous."""
    s_count, t_count, a_count, x_count, r_count = problem.shape
    cell = t_count * a_count * x_count * r_count            # (t,a,x,r) block per locality
    st = t_count                                             # (locality,time)
    sxr = t_count * x_count * r_count                        # births: (locality,time,sex,race)

    def _cell(arr: np.ndarray | None) -> np.ndarray | None:
        return None if arr is None else arr[s0 * cell:s1 * cell]

    def _st(arr: np.ndarray | None) -> np.ndarray | None:
        return None if arr is None else arr[s0 * st:s1 * st]

    def _sxr(arr: np.ndarray | None) -> np.ndarray | None:
        return None if arr is None else arr[s0 * sxr:s1 * sxr]

    return PopulationTensorProblem(
        shape=(s1 - s0, t_count, a_count, x_count, r_count),
        anchors=_cell(problem.anchors),
        hard_anchor_mask=_cell(problem.hard_anchor_mask),
        mode=problem.mode,
        births=_sxr(problem.births),
        death_rates=_cell(problem.death_rates),
        sim_deaths=_cell(problem.sim_deaths),
        race_composition_prior=_cell(problem.race_composition_prior),
        closure_totals=_st(problem.closure_totals),
        migration_totals=None,  # separability precondition (asserted by the caller)
        migration_locality_totals=_st(problem.migration_locality_totals),
        migration_bounds=_cell(problem.migration_bounds),
        initial_population=_cell(problem.initial_population),
        initial_migration=_cell(problem.initial_migration),
        weights=problem.weights,
    )


def _locality_separable(problem: PopulationTensorProblem) -> bool:
    """The only loss term that couples localities is the per-stratum migration enclosure
    (``migration_totals``, shape (t,a,x,r), summed across localities). Absent it, the objective
    decomposes exactly by locality. The SIDRA denominator build never sets it."""
    return problem.migration_totals is None


def solve_population_tensor_blocked(
    problem: PopulationTensorProblem,
    *,
    solver_id: str | None = None,
    max_iterations: int = 5_000,
    tolerance: float = 1e-5,
    block_target_cells: int = _BLOCK_TARGET_CELLS,
) -> PopulationOptimizationResult:
    """Solve the population tensor in locality-blocks so peak memory is O(block), not O(national)
    (§V.1(b)). Exact when the problem is locality-separable; otherwise solves whole. Each block
    re-selects its solver by its own (small) cell count, so blocks take the fast dense path."""
    s_count = problem.shape[0]
    per_locality = max(1, problem.n_cells // max(1, s_count))
    block_localities = max(1, block_target_cells // per_locality)
    if not _locality_separable(problem) or s_count <= block_localities:
        return solve_population_tensor_problem(
            problem, solver_id=solver_id, max_iterations=max_iterations, tolerance=tolerance,
        )

    pops: list[np.ndarray] = []
    migs: list[np.ndarray] = []
    converged_all = True
    iters_max = 0
    final_objective = 0.0
    initial_objective = 0.0
    worst_grad = 0.0
    for s0 in range(0, s_count, block_localities):
        s1 = min(s0 + block_localities, s_count)
        sub = _slice_localities(problem, s0, s1)
        # solver_id=None → each block re-selects by its own cell count (small → dense fast path).
        res = solve_population_tensor_problem(
            sub, solver_id=None, max_iterations=max_iterations, tolerance=tolerance,
        )
        pops.append(np.asarray(res.population, dtype=np.float64))
        migs.append(np.asarray(res.migration, dtype=np.float64))
        converged_all = converged_all and res.telemetry.converged
        iters_max = max(iters_max, res.telemetry.iterations)
        final_objective += res.telemetry.final_objective
        initial_objective += res.telemetry.initial_objective
        worst_grad = max(worst_grad, res.telemetry.projected_gradient_norm)

    population = np.concatenate(pops) if pops else np.zeros(0)
    migration = np.concatenate(migs) if migs else np.zeros(0)
    telemetry = PopulationSolverTelemetry(
        converged=converged_all,
        iterations=iters_max,
        initial_objective=initial_objective,
        final_objective=final_objective,
        projected_gradient_norm=worst_grad,
        relative_objective_change=0.0,
        step_size=0.0,
        objective_terms={"blocked_localities": float(block_localities), "n_blocks": float(len(pops))},
    )
    return PopulationOptimizationResult(tuple(float(x) for x in population), tuple(float(x) for x in migration), telemetry)


__all__ = ["solve_population_tensor_problem", "solve_population_tensor_blocked", "dense_national_abort_check"]
