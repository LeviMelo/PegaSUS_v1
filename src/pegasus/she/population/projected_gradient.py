from __future__ import annotations

import math
from dataclasses import dataclass

from pegasus.she.population.loss import evaluate_population_loss, validate_population_problem
from pegasus.she.population.schema import PopulationSolverTelemetry, PopulationTensorProblem


@dataclass(frozen=True)
class PopulationOptimizationResult:
    population: tuple[float, ...]
    migration: tuple[float, ...]
    telemetry: PopulationSolverTelemetry


def _simplex_projection(values: list[float], total: float) -> list[float]:
    if total < 0:
        raise ValueError("Closure totals must be nonnegative.")
    if not values:
        if abs(total) > 1e-9:
            raise ValueError("Closure total cannot be satisfied without free cells.")
        return []
    ordered = sorted(values, reverse=True)
    cumulative = 0.0
    rho = 0
    for j, value in enumerate(ordered, start=1):
        cumulative += value
        if value - (cumulative - total) / j > 0:
            rho = j
    theta = (sum(ordered[:rho]) - total) / rho if rho else 0.0
    return [max(value - theta, 0.0) for value in values]


def _project_population(problem: PopulationTensorProblem, values: list[float]) -> list[float]:
    projected = [max(value, 0.0) for value in values]
    hard = problem.hard_anchor_mask or (False,) * problem.n_cells
    for i, locked in enumerate(hard):
        if locked:
            anchor = problem.anchors[i]
            if anchor is None or anchor < 0:
                raise ValueError("Hard population anchors must be present and nonnegative.")
            projected[i] = anchor

    if problem.closure_totals is None:
        return projected
    s_count, t_count, a_count, x_count, r_count = problem.shape
    group_size = a_count * x_count * r_count
    for s in range(s_count):
        for t in range(t_count):
            total = problem.closure_totals[s * t_count + t]
            if total is None:
                continue
            start = (s * t_count + t) * group_size
            indices = list(range(start, start + group_size))
            fixed = [i for i in indices if hard[i]]
            free = [i for i in indices if not hard[i]]
            remainder = total - sum(projected[i] for i in fixed)
            if remainder < -1e-8:
                raise ValueError("Hard population anchors exceed a closure total.")
            free_values = _simplex_projection([projected[i] for i in free], max(remainder, 0.0))
            for i, value in zip(free, free_values, strict=True):
                projected[i] = value
    return projected


def _migration_bounds(problem: PopulationTensorProblem) -> tuple[float, ...]:
    if problem.migration_bounds is not None:
        if any(bound < 0 or not math.isfinite(bound) for bound in problem.migration_bounds):
            raise ValueError("Migration bounds must be finite and nonnegative.")
        return problem.migration_bounds
    fallback = max((anchor or 0.0 for anchor in problem.anchors), default=0.0)
    if problem.closure_totals is not None:
        fallback = max(fallback, max((value or 0.0 for value in problem.closure_totals), default=0.0))
    return (max(fallback, 1.0),) * problem.n_cells


def _bounded_sum_projection(values: list[float], bounds: list[float], total: float) -> list[float]:
    if total < -sum(bounds) - 1e-9 or total > sum(bounds) + 1e-9:
        raise ValueError("Migration enclosure total is infeasible under configured bounds.")
    low = min(value - bound for value, bound in zip(values, bounds, strict=True)) - abs(total) - 1.0
    high = max(value + bound for value, bound in zip(values, bounds, strict=True)) + abs(total) + 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        projected = [min(max(value - midpoint, -bound), bound) for value, bound in zip(values, bounds, strict=True)]
        if sum(projected) > total:
            low = midpoint
        else:
            high = midpoint
    midpoint = (low + high) / 2.0
    return [min(max(value - midpoint, -bound), bound) for value, bound in zip(values, bounds, strict=True)]


def _project_migration(problem: PopulationTensorProblem, values: list[float], bounds: tuple[float, ...]) -> list[float]:
    projected = [min(max(value, -bound), bound) for value, bound in zip(values, bounds, strict=True)]
    if problem.migration_totals is None:
        return projected
    s_count, t_count, a_count, x_count, r_count = problem.shape
    for t in range(t_count):
        for a in range(a_count):
            for x in range(x_count):
                for r in range(r_count):
                    total_index = (((t * a_count) + a) * x_count + x) * r_count + r
                    total = problem.migration_totals[total_index]
                    if total is None:
                        continue
                    indices = [((((s * t_count) + t) * a_count + a) * x_count + x) * r_count + r for s in range(s_count)]
                    group = _bounded_sum_projection([projected[i] for i in indices], [bounds[i] for i in indices], total)
                    for i, value in zip(indices, group, strict=True):
                        projected[i] = value
    return projected


def _initial_population(problem: PopulationTensorProblem) -> list[float]:
    if problem.initial_population is not None:
        return _project_population(problem, list(problem.initial_population))
    values = [anchor if anchor is not None else 0.0 for anchor in problem.anchors]
    if problem.closure_totals is not None:
        s_count, t_count, a_count, x_count, r_count = problem.shape
        group_size = a_count * x_count * r_count
        for s in range(s_count):
            for t in range(t_count):
                total = problem.closure_totals[s * t_count + t]
                if total is None:
                    continue
                start = (s * t_count + t) * group_size
                if not any(problem.anchors[i] is not None for i in range(start, start + group_size)):
                    for i in range(start, start + group_size):
                        values[i] = total / group_size
    return _project_population(problem, values)


def solve_projected_gradient_small(
    problem: PopulationTensorProblem,
    *,
    max_iterations: int = 5_000,
    tolerance: float = 1e-5,
    initial_step_size: float = 1.0,
) -> PopulationOptimizationResult:
    validate_population_problem(problem)
    if max_iterations <= 0 or tolerance <= 0 or initial_step_size <= 0:
        raise ValueError("Solver controls must be positive.")
    population = _initial_population(problem)
    bounds = _migration_bounds(problem)
    migration = _project_migration(problem, list(problem.initial_migration or (0.0,) * problem.n_cells), bounds)
    evaluation = evaluate_population_loss(problem, population, migration)
    initial_objective = evaluation.total
    previous_objective = initial_objective
    step_size = initial_step_size
    relative_change = 0.0
    projected_norm = math.inf
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        unit_population = _project_population(
            problem,
            [value - gradient for value, gradient in zip(population, evaluation.population_gradient, strict=True)],
        )
        unit_migration = _project_migration(
            problem,
            [value - gradient for value, gradient in zip(migration, evaluation.migration_gradient, strict=True)],
            bounds,
        )
        projected_norm = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(population, unit_population, strict=True))
            + sum((a - b) ** 2 for a, b in zip(migration, unit_migration, strict=True))
        )
        if projected_norm <= tolerance:
            converged = True
            iterations = iteration - 1
            break

        trial_step = step_size
        accepted = False
        for _ in range(30):
            trial_population = _project_population(
                problem,
                [value - trial_step * gradient for value, gradient in zip(population, evaluation.population_gradient, strict=True)],
            )
            trial_migration = _project_migration(
                problem,
                [value - trial_step * gradient for value, gradient in zip(migration, evaluation.migration_gradient, strict=True)],
                bounds,
            )
            trial = evaluate_population_loss(problem, trial_population, trial_migration)
            if trial.total <= evaluation.total + 1e-12:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            iterations = iteration
            break

        population = trial_population
        migration = trial_migration
        evaluation = trial
        relative_change = abs(previous_objective - evaluation.total) / max(abs(previous_objective), 1.0)
        previous_objective = evaluation.total
        step_size = min(trial_step * 1.2, initial_step_size)
        iterations = iteration

    telemetry = PopulationSolverTelemetry(
        converged=converged,
        iterations=iterations,
        initial_objective=initial_objective,
        final_objective=evaluation.total,
        projected_gradient_norm=projected_norm,
        relative_objective_change=relative_change,
        step_size=step_size,
        objective_terms=evaluation.terms,
    )
    return PopulationOptimizationResult(tuple(population), tuple(migration), telemetry)
