"""Sparse population solver facade with scale and memory proofs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pegasus.compute.memory import MemoryPreflight, preflight_memory
from pegasus.she.reconstruction.block_coordinate import solve_population_block_coordinate
from pegasus.she.reconstruction.loss import evaluate_population_loss, validate_population_problem
from pegasus.she.reconstruction.projected_gradient import (
    PopulationOptimizationResult,
    _initial_population,
    _migration_bounds,
    _project_migration,
    _project_population,
)
from pegasus.she.reconstruction.schema import PopulationSolverTelemetry, PopulationTensorProblem
from pegasus.she.reconstruction.state_space import PopulationStateSpace, build_population_state_space


@dataclass(frozen=True)
class SparsePopulationPlan:
    solver_id: str
    state_space: PopulationStateSpace
    memory: MemoryPreflight
    sparse_jacobian: bool = True

    def as_manifest(self) -> dict[str, Any]:
        return {
            "solver_id": self.solver_id,
            "sparse_jacobian": self.sparse_jacobian,
            "state_space": self.state_space.as_manifest(),
            "memory": self.memory.as_manifest(),
        }


def plan_sparse_population_solver(
    problem: PopulationTensorProblem,
    *,
    solver_id: str = "sparse_block_coordinate_v1",
    available_bytes: int | None = None,
    max_memory_fraction: float = 0.8,
) -> SparsePopulationPlan:
    state_space = build_population_state_space(problem)
    # Two float64 iterates, two gradients, and conservative edge/index overhead.
    estimated_bytes = problem.n_cells * 8 * 6 + (len(state_space.aging_edges) + len(state_space.birth_edges)) * 24
    memory = preflight_memory(
        estimated_bytes,
        available_bytes=available_bytes,
        max_fraction=max_memory_fraction,
        memory_kind="ram",
    )
    return SparsePopulationPlan(solver_id=solver_id, state_space=state_space, memory=memory)


def solve_sparse_population(
    problem: PopulationTensorProblem,
    *,
    solver_id: str = "sparse_block_coordinate_v1",
    max_iterations: int = 2_000,
    tolerance: float = 1e-5,
    available_bytes: int | None = None,
) -> tuple[PopulationOptimizationResult, SparsePopulationPlan]:
    plan = plan_sparse_population_solver(problem, solver_id=solver_id, available_bytes=available_bytes)
    result = solve_population_block_coordinate(problem, max_iterations=max_iterations, tolerance=tolerance)
    return result, plan


def _norm2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def solve_population_admm(
    problem: PopulationTensorProblem,
    *,
    solver_id: str = "sim_informed_sparse_admm_v1",
    max_iterations: int = 2_000,
    tolerance: float = 1e-5,
    rho: float = 1.0,
    available_bytes: int | None = None,
) -> tuple[PopulationOptimizationResult, SparsePopulationPlan]:
    """Executable split-projection ADMM backend for the population objective.

    The smooth objective is evaluated with the analytic gradient from
    ``loss.py``. The constrained split variables are projected through the same
    nonnegativity, hard-anchor, closure-total, migration-bound, and
    migration-total operators used by the projected-gradient solver.
    """
    validate_population_problem(problem)
    if max_iterations <= 0 or tolerance <= 0 or rho <= 0:
        raise ValueError("ADMM controls must be positive.")
    plan = plan_sparse_population_solver(problem, solver_id=solver_id, available_bytes=available_bytes)
    bounds = _migration_bounds(problem)

    population = _initial_population(problem)
    _init_mig = problem.initial_migration if problem.initial_migration is not None else [0.0] * problem.n_cells
    migration = _project_migration(problem, list(_init_mig), bounds)
    z_population = list(population)
    z_migration = list(migration)
    u_population = [0.0] * problem.n_cells
    u_migration = [0.0] * problem.n_cells
    evaluation = evaluate_population_loss(problem, population, migration)
    initial_objective = evaluation.total
    previous_objective = initial_objective
    step_size = 1.0 / max(rho, 1.0)
    converged = False
    relative_change = 0.0
    residual_norm = math.inf
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        population = [
            p - step_size * (g + rho * (p - z + u))
            for p, g, z, u in zip(population, evaluation.population_gradient, z_population, u_population, strict=True)
        ]
        migration = [
            m - step_size * (g + rho * (m - z + u))
            for m, g, z, u in zip(migration, evaluation.migration_gradient, z_migration, u_migration, strict=True)
        ]

        z_population_old = z_population
        z_migration_old = z_migration
        z_population = _project_population(problem, [p + u for p, u in zip(population, u_population, strict=True)])
        z_migration = _project_migration(problem, [m + u for m, u in zip(migration, u_migration, strict=True)], bounds)
        u_population = [u + p - z for u, p, z in zip(u_population, population, z_population, strict=True)]
        u_migration = [u + m - z for u, m, z in zip(u_migration, migration, z_migration, strict=True)]

        population = list(z_population)
        migration = list(z_migration)
        evaluation = evaluate_population_loss(problem, population, migration)

        primal = (population[i] - z_population[i] for i in range(problem.n_cells))
        migration_primal = (migration[i] - z_migration[i] for i in range(problem.n_cells))
        dual = (rho * (z_population[i] - z_population_old[i]) for i in range(problem.n_cells))
        migration_dual = (rho * (z_migration[i] - z_migration_old[i]) for i in range(problem.n_cells))
        residual_norm = _norm2(list(primal) + list(migration_primal) + list(dual) + list(migration_dual))
        relative_change = abs(previous_objective - evaluation.total) / max(abs(previous_objective), 1.0)
        previous_objective = evaluation.total
        iterations = iteration
        if residual_norm <= tolerance or relative_change <= tolerance:
            converged = True
            break

    telemetry = PopulationSolverTelemetry(
        converged=converged,
        iterations=iterations,
        initial_objective=initial_objective,
        final_objective=evaluation.total,
        projected_gradient_norm=residual_norm,
        relative_objective_change=relative_change,
        step_size=step_size,
        objective_terms=evaluation.terms | {"admm_rho": rho},
    )
    return PopulationOptimizationResult(tuple(population), tuple(migration), telemetry), plan
