"""Analytic population tensor objective and sparse gradient.

Typed formula contract
----------------------
Inputs are flat float64 vectors for population ``P`` and migration ``eta`` on
shape ``(S, T, A, X, R)``. Optional observations use ``None`` for missingness.
The objective implements anchor fit, cohort aging, newborn entry, SIM death
prior, migration second differences, ILR race composition, and age second differences. Independent
mode requires a zero death weight. Outputs are a scalar loss, named component
losses, and analytic gradients with the same flat support. Nonnegativity,
closure totals, hard anchors, and migration bounds are enforced by projection
in the solver. ILR logs use epsilon ``1e-12`` for structural-zero stability. Invalid shapes, negative
weights, non-finite values, or SIM priors in independent mode fail explicitly.
Non-convergence downgrades state and emits a warning in the solver result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pegasus.she.population.schema import PopulationTensorProblem


@dataclass(frozen=True)
class PopulationLossEvaluation:
    total: float
    terms: dict[str, float]
    population_gradient: tuple[float, ...]
    migration_gradient: tuple[float, ...]


def validate_population_problem(problem: PopulationTensorProblem) -> None:
    if len(problem.shape) != 5 or any(dimension <= 0 for dimension in problem.shape):
        raise ValueError("Population tensor shape must contain five positive dimensions.")
    n = problem.n_cells
    if len(problem.anchors) != n:
        raise ValueError(f"anchors length {len(problem.anchors)} does not match n_cells {n}.")
    for name in (
        "hard_anchor_mask",
        "death_rates",
        "sim_deaths",
        "race_composition_prior",
        "migration_bounds",
        "initial_population",
        "initial_migration",
    ):
        value = getattr(problem, name)
        if value is not None and len(value) != n:
            raise ValueError(f"{name} length {len(value)} does not match n_cells {n}.")
    s_count, t_count, _, x_count, r_count = problem.shape
    if problem.births is not None and len(problem.births) != s_count * t_count * x_count * r_count:
        raise ValueError("births must have shape (locality, time, sex, race).")
    if problem.closure_totals is not None and len(problem.closure_totals) != s_count * t_count:
        raise ValueError("closure_totals must have shape (locality, time).")
    if problem.migration_totals is not None and len(problem.migration_totals) != t_count * problem.shape[2] * x_count * r_count:
        raise ValueError("migration_totals must have shape (time, age, sex, race).")
    weights = problem.weights
    if any(value < 0 or not math.isfinite(value) for value in vars(weights).values()):
        raise ValueError("Population objective weights must be finite and nonnegative.")
    if problem.mode == "independent_denominator" and weights.death != 0:
        raise ValueError("Independent denominator mode requires death weight lambda_D=0.")
    if problem.mode == "sim_informed_denominator" and weights.death <= 0:
        raise ValueError("SIM-informed denominator mode requires death weight lambda_D>0.")


def _index(shape: tuple[int, int, int, int, int], s: int, t: int, a: int, x: int, r: int) -> int:
    _, t_count, a_count, x_count, r_count = shape
    return ((((s * t_count) + t) * a_count + a) * x_count + x) * r_count + r


def _birth_index(shape: tuple[int, int, int, int, int], s: int, t: int, x: int, r: int) -> int:
    _, t_count, _, x_count, r_count = shape
    return (((s * t_count) + t) * x_count + x) * r_count + r


def _ilr(values: list[float], *, epsilon: float = 1e-12) -> tuple[list[float], list[list[float]]]:
    count = len(values)
    if count < 2:
        return [], []
    positive = [max(value, epsilon) for value in values]
    total = sum(positive)
    logs = [math.log(value / total) for value in positive]
    basis: list[list[float]] = []
    coordinates: list[float] = []
    for j in range(count - 1):
        scale = math.sqrt((j + 1) / (j + 2))
        row = [0.0] * count
        for r in range(j + 1):
            row[r] = scale / (j + 1)
        row[j + 1] = -scale
        basis.append(row)
        coordinates.append(sum(coefficient * value for coefficient, value in zip(row, logs, strict=True)))
    return coordinates, basis


def evaluate_population_loss(
    problem: PopulationTensorProblem,
    population: tuple[float, ...] | list[float],
    migration: tuple[float, ...] | list[float],
) -> PopulationLossEvaluation:
    validate_population_problem(problem)
    n = problem.n_cells
    if len(population) != n or len(migration) != n:
        raise ValueError("Population and migration vectors must match problem n_cells.")
    if any(not math.isfinite(value) for value in (*population, *migration)):
        raise ValueError("Population objective received a non-finite iterate.")

    gp = [0.0] * n
    gm = [0.0] * n
    terms = {name: 0.0 for name in ("anchor", "aging", "birth", "death", "migration", "race", "age_smooth")}
    w = problem.weights
    shape = problem.shape
    s_count, t_count, a_count, x_count, r_count = shape

    for i, anchor in enumerate(problem.anchors):
        if anchor is None:
            continue
        residual = population[i] - anchor
        terms["anchor"] += w.anchor * residual * residual
        gp[i] += 2.0 * w.anchor * residual

    death_rates = problem.death_rates or (None,) * n
    for s in range(s_count):
        for t in range(1, t_count):
            for x in range(x_count):
                for r in range(r_count):
                    for a in range(1, a_count):
                        current = _index(shape, s, t, a, x, r)
                        prior_age = a - 1
                        prior = _index(shape, s, t - 1, prior_age, x, r)
                        eta = _index(shape, s, t - 1, a, x, r)
                        survival = 1.0 - (death_rates[prior] or 0.0)
                        expected = population[prior] * survival + migration[eta]
                        contributors = [(prior, survival)]
                        if a == a_count - 1 and a_count > 1:
                            terminal = _index(shape, s, t - 1, a, x, r)
                            terminal_survival = 1.0 - (death_rates[terminal] or 0.0)
                            expected += population[terminal] * terminal_survival
                            contributors.append((terminal, terminal_survival))
                        residual = population[current] - expected
                        terms["aging"] += w.aging * residual * residual
                        gp[current] += 2.0 * w.aging * residual
                        for contributor, coefficient in contributors:
                            gp[contributor] -= 2.0 * w.aging * residual * coefficient
                        gm[eta] -= 2.0 * w.aging * residual

                    if problem.births is not None:
                        current = _index(shape, s, t, 0, x, r)
                        eta = _index(shape, s, t - 1, 0, x, r)
                        birth = problem.births[_birth_index(shape, s, t - 1, x, r)]
                        if birth is not None:
                            residual = population[current] - birth - migration[eta]
                            terms["birth"] += w.birth * residual * residual
                            gp[current] += 2.0 * w.birth * residual
                            gm[eta] -= 2.0 * w.birth * residual

    if problem.sim_deaths is not None and w.death > 0:
        for i, deaths in enumerate(problem.sim_deaths):
            rate = death_rates[i]
            if deaths is None or rate is None:
                continue
            residual = rate * population[i] - deaths
            terms["death"] += w.death * residual * residual
            gp[i] += 2.0 * w.death * residual * rate

    if problem.race_composition_prior is not None and w.race > 0:
        prior = problem.race_composition_prior
        for s in range(s_count):
            for t in range(t_count):
                for a in range(a_count):
                    for x in range(x_count):
                        indices = [_index(shape, s, t, a, x, r) for r in range(r_count)]
                        if any(prior[i] is None for i in indices):
                            continue
                        observed = [population[i] for i in indices]
                        if sum(observed) <= 1e-12:
                            continue
                        target = [float(prior[i]) for i in indices]
                        observed_ilr, basis = _ilr(observed)
                        target_ilr, _ = _ilr(target)
                        residuals = [value - target_value for value, target_value in zip(observed_ilr, target_ilr, strict=True)]
                        terms["race"] += w.race * sum(residual * residual for residual in residuals)
                        for r, i in enumerate(indices):
                            derivative = sum(residual * row[r] for residual, row in zip(residuals, basis, strict=True))
                            gp[i] += 2.0 * w.race * derivative / max(population[i], 1e-12)

    if w.migration > 0 and t_count >= 3:
        for s in range(s_count):
            for t in range(2, t_count):
                for a in range(a_count):
                    for x in range(x_count):
                        for r in range(r_count):
                            i2 = _index(shape, s, t, a, x, r)
                            i1 = _index(shape, s, t - 1, a, x, r)
                            i0 = _index(shape, s, t - 2, a, x, r)
                            residual = migration[i2] - 2.0 * migration[i1] + migration[i0]
                            terms["migration"] += w.migration * residual * residual
                            gm[i2] += 2.0 * w.migration * residual
                            gm[i1] -= 4.0 * w.migration * residual
                            gm[i0] += 2.0 * w.migration * residual

    if w.age_smooth > 0 and a_count >= 3:
        for s in range(s_count):
            for t in range(t_count):
                for a in range(2, a_count):
                    for x in range(x_count):
                        for r in range(r_count):
                            i2 = _index(shape, s, t, a, x, r)
                            i1 = _index(shape, s, t, a - 1, x, r)
                            i0 = _index(shape, s, t, a - 2, x, r)
                            residual = population[i2] - 2.0 * population[i1] + population[i0]
                            terms["age_smooth"] += w.age_smooth * residual * residual
                            gp[i2] += 2.0 * w.age_smooth * residual
                            gp[i1] -= 4.0 * w.age_smooth * residual
                            gp[i0] += 2.0 * w.age_smooth * residual

    return PopulationLossEvaluation(
        total=sum(terms.values()),
        terms=terms,
        population_gradient=tuple(gp),
        migration_gradient=tuple(gm),
    )
