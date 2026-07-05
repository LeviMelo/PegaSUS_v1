"""Memory-bounded block-coordinate population optimizer."""

from __future__ import annotations

import math

from pegasus.she.reconstruction.loss import evaluate_population_loss, validate_population_problem
from pegasus.she.reconstruction.projected_gradient import (
    PopulationOptimizationResult,
    _initial_population,
    _migration_bounds,
    _project_migration,
    _project_population,
)
from pegasus.she.reconstruction.schema import PopulationSolverTelemetry, PopulationTensorProblem


def solve_population_block_coordinate(
    problem: PopulationTensorProblem,
    *,
    max_iterations: int = 2_000,
    tolerance: float = 1e-5,
    initial_step_size: float = 0.25,
) -> PopulationOptimizationResult:
    validate_population_problem(problem)
    if max_iterations <= 0 or tolerance <= 0 or initial_step_size <= 0:
        raise ValueError("Solver controls must be positive.")
    population = _initial_population(problem)
    bounds = _migration_bounds(problem)
    _init_mig = problem.initial_migration if problem.initial_migration is not None else [0.0] * problem.n_cells
    migration = _project_migration(problem, list(_init_mig), bounds)
    evaluation = evaluate_population_loss(problem, population, migration)
    initial = evaluation.total
    previous = initial
    relative_change = math.inf
    projected_norm = math.inf
    step = initial_step_size
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        trial_population = _project_population(
            problem,
            [value - step * gradient for value, gradient in zip(population, evaluation.population_gradient, strict=True)],
        )
        population_eval = evaluate_population_loss(problem, trial_population, migration)
        if population_eval.total > evaluation.total + 1e-12:
            step *= 0.5
            if step < 1e-12:
                break
            continue
        trial_migration = _project_migration(
            problem,
            [value - step * gradient for value, gradient in zip(migration, population_eval.migration_gradient, strict=True)],
            bounds,
        )
        trial = evaluate_population_loss(problem, trial_population, trial_migration)
        delta = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(population, trial_population, strict=True))
            + sum((a - b) ** 2 for a, b in zip(migration, trial_migration, strict=True))
        )
        population, migration, evaluation = trial_population, trial_migration, trial
        relative_change = abs(previous - evaluation.total) / max(abs(previous), 1.0)
        previous = evaluation.total
        projected_norm = delta / max(step, 1e-12)
        iterations = iteration
        if projected_norm <= tolerance or relative_change <= tolerance:
            converged = True
            break
        step = min(step * 1.05, initial_step_size)
    telemetry = PopulationSolverTelemetry(
        converged=converged,
        iterations=iterations,
        initial_objective=initial,
        final_objective=evaluation.total,
        projected_gradient_norm=projected_norm,
        relative_objective_change=relative_change,
        step_size=step,
        objective_terms=evaluation.terms,
    )
    return PopulationOptimizationResult(tuple(population), tuple(migration), telemetry)
