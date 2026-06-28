"""Sparse population state-space indexing and transition plan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

from pegasus.she.population.projected_gradient import PopulationOptimizationResult, _initial_population, _project_population
from pegasus.she.population.schema import PopulationSolverTelemetry
from pegasus.she.population.schema import PopulationTensorProblem


@dataclass(frozen=True)
class PopulationStateSpace:
    shape: tuple[int, int, int, int, int]
    anchor_indices: tuple[int, ...]
    aging_edges: tuple[tuple[int, int], ...]
    birth_edges: tuple[tuple[int, int], ...]
    migration_indices: tuple[int, ...]
    race_groups: tuple[tuple[int, ...], ...]
    closure_groups: tuple[tuple[int, ...], ...]

    @property
    def n_cells(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def as_manifest(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "n_cells": self.n_cells,
            "anchor_count": len(self.anchor_indices),
            "aging_edge_count": len(self.aging_edges),
            "birth_edge_count": len(self.birth_edges),
            "migration_cell_count": len(self.migration_indices),
            "race_group_count": len(self.race_groups),
            "closure_group_count": len(self.closure_groups),
        }


@dataclass(frozen=True)
class PopulationStateSpaceSmootherResult:
    state_space: PopulationStateSpace
    result: PopulationOptimizationResult
    process_variance: float
    observation_variance: float

    def as_manifest(self) -> dict[str, Any]:
        payload = self.state_space.as_manifest()
        payload.update(
            {
                "solver_backend": "state_space_smoother_reduced_rts",
                "process_variance": self.process_variance,
                "observation_variance": self.observation_variance,
                "telemetry": self.result.telemetry.as_manifest(),
            }
        )
        return payload


def flat_index(shape: tuple[int, int, int, int, int], s: int, t: int, a: int, x: int, r: int) -> int:
    _, periods, ages, sexes, races = shape
    return ((((s * periods) + t) * ages + a) * sexes + x) * races + r


def iter_coordinates(shape: tuple[int, int, int, int, int]) -> Iterator[tuple[int, int, int, int, int]]:
    localities, periods, ages, sexes, races = shape
    for s in range(localities):
        for t in range(periods):
            for a in range(ages):
                for x in range(sexes):
                    for r in range(races):
                        yield s, t, a, x, r


def build_population_state_space(problem: PopulationTensorProblem) -> PopulationStateSpace:
    shape = problem.shape
    localities, periods, ages, sexes, races = shape
    anchors = tuple(i for i, value in enumerate(problem.anchors) if value is not None)
    aging: list[tuple[int, int]] = []
    births: list[tuple[int, int]] = []
    race_groups: list[tuple[int, ...]] = []
    closures: list[tuple[int, ...]] = []
    for s in range(localities):
        for t in range(periods):
            closures.append(tuple(flat_index(shape, s, t, a, x, r) for a in range(ages) for x in range(sexes) for r in range(races)))
            for a in range(ages):
                for x in range(sexes):
                    race_groups.append(tuple(flat_index(shape, s, t, a, x, r) for r in range(races)))
            if t == 0:
                continue
            for x in range(sexes):
                for r in range(races):
                    births.append((flat_index(shape, s, t - 1, 0, x, r), flat_index(shape, s, t, 0, x, r)))
                    for a in range(1, ages):
                        aging.append((flat_index(shape, s, t - 1, a - 1, x, r), flat_index(shape, s, t, a, x, r)))
    return PopulationStateSpace(
        shape=shape,
        anchor_indices=anchors,
        aging_edges=tuple(aging),
        birth_edges=tuple(births),
        migration_indices=tuple(range(problem.n_cells)),
        race_groups=tuple(race_groups),
        closure_groups=tuple(closures),
    )


def _measurement(problem: PopulationTensorProblem, index: int) -> float | None:
    value = problem.anchors[index]
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _smooth_path(
    observations: list[float | None],
    *,
    initial: float,
    process_variance: float,
    observation_variance: float,
) -> list[float]:
    state = float(max(initial, 0.0))
    covariance = observation_variance
    filtered: list[float] = []
    covariances: list[float] = []
    predicted: list[float] = []
    predicted_covariances: list[float] = []
    for observation in observations:
        predicted.append(state)
        pred_cov = covariance + process_variance
        predicted_covariances.append(pred_cov)
        if observation is not None:
            gain = pred_cov / (pred_cov + observation_variance)
            state = state + gain * (observation - state)
            covariance = (1.0 - gain) * pred_cov
        else:
            covariance = pred_cov
        filtered.append(max(state, 0.0))
        covariances.append(max(covariance, 1e-12))

    smoothed = filtered[:]
    for i in range(len(smoothed) - 2, -1, -1):
        gain = covariances[i] / max(predicted_covariances[i + 1], 1e-12)
        smoothed[i] = max(filtered[i] + gain * (smoothed[i + 1] - predicted[i + 1]), 0.0)
    return smoothed


def solve_population_state_space_smoother(
    problem: PopulationTensorProblem,
    *,
    process_variance: float = 1.0,
    observation_variance: float = 4.0,
) -> PopulationStateSpaceSmootherResult:
    """Reduced Kalman/RTS-style smoother over cohort paths.

    This backend is intentionally limited to the reduced UF/region path named in
    the MSD. It preserves the same population projection constraints as the
    gradient solvers, but fills missing cells by smoothing along cohort-aging
    trajectories rather than treating the state-space module as indexing only.
    """
    if process_variance <= 0 or observation_variance <= 0:
        raise ValueError("State-space variances must be positive.")
    state_space = build_population_state_space(problem)
    s_count, t_count, a_count, x_count, r_count = problem.shape
    initial = _initial_population(problem)
    values = initial[:]

    for s in range(s_count):
        for x in range(x_count):
            for r in range(r_count):
                for cohort_start_age in range(a_count):
                    path: list[int] = []
                    for t in range(t_count):
                        age = cohort_start_age + t
                        if age >= a_count:
                            break
                        path.append(flat_index(problem.shape, s, t, age, x, r))
                    if path:
                        observations = [_measurement(problem, idx) for idx in path]
                        smoothed = _smooth_path(
                            observations,
                            initial=values[path[0]],
                            process_variance=process_variance,
                            observation_variance=observation_variance,
                        )
                        for idx, value in zip(path, smoothed, strict=True):
                            values[idx] = value

                newborn_path = [flat_index(problem.shape, s, t, 0, x, r) for t in range(t_count)]
                observations = [_measurement(problem, idx) for idx in newborn_path]
                if problem.births is not None:
                    birth_shape = (s_count, t_count, x_count, r_count)
                    for t in range(t_count):
                        birth_idx = (((s * birth_shape[1]) + t) * birth_shape[2] + x) * birth_shape[3] + r
                        if observations[t] is None and problem.births[birth_idx] is not None:
                            observations[t] = float(problem.births[birth_idx] or 0.0)
                smoothed = _smooth_path(
                    observations,
                    initial=values[newborn_path[0]],
                    process_variance=process_variance,
                    observation_variance=observation_variance,
                )
                for idx, value in zip(newborn_path, smoothed, strict=True):
                    values[idx] = value

    population = _project_population(problem, values)
    migration = tuple(0.0 for _ in range(problem.n_cells))
    delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(initial, population, strict=True)))
    telemetry = PopulationSolverTelemetry(
        converged=True,
        iterations=1,
        initial_objective=0.0,
        final_objective=0.0,
        projected_gradient_norm=delta,
        relative_objective_change=0.0,
        step_size=1.0,
        objective_terms={"state_space_smoothing": delta},
    )
    return PopulationStateSpaceSmootherResult(
        state_space=state_space,
        result=PopulationOptimizationResult(tuple(population), migration, telemetry),
        process_variance=process_variance,
        observation_variance=observation_variance,
    )
