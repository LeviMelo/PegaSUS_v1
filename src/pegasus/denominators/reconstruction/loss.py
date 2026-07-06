"""Analytic population tensor objective and sparse gradient.

Typed formula contract
----------------------
Inputs are flat float64 vectors for population ``P`` and migration ``eta`` on
shape ``(S, T, A, X, R)``. Optional observations use ``None`` for missingness.
The objective implements anchor fit, cohort aging, newborn entry, SIM death
prior, migration second differences, per-locality net-migration total residual,
ILR race composition, and age second differences. Independent
mode requires a zero death weight. Outputs are a scalar loss, named component
losses, and analytic gradients with the same flat support. Nonnegativity,
closure totals, hard anchors, and migration bounds are enforced by projection
in the solver. ILR logs use epsilon ``1e-12`` for structural-zero stability. Invalid shapes, negative
weights, non-finite values, or SIM priors in independent mode fail explicitly.
Non-convergence downgrades state and emits a warning in the solver result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pegasus.denominators.reconstruction.schema import PopulationTensorProblem


@dataclass(frozen=True)
class PopulationLossEvaluation:
    # gradients are numpy arrays, NOT Python float tuples: the tuple(float(x) for x in gp) round-trip
    # was ~99% of a loss evaluation (~110ms→~1ms at block scale, and the loss runs 2x per SPG iter).
    total: float
    terms: dict[str, float]
    population_gradient: np.ndarray
    migration_gradient: np.ndarray


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
    if problem.migration_locality_totals is not None and len(problem.migration_locality_totals) != s_count * t_count:
        raise ValueError("migration_locality_totals must have shape (locality, time).")
    if problem.migration_totals is not None and len(problem.migration_totals) != t_count * problem.shape[2] * x_count * r_count:
        raise ValueError("migration_totals must have shape (time, age, sex, race).")
    weights = problem.weights
    if any(value < 0 or not math.isfinite(value) for value in vars(weights).values()):
        raise ValueError("Population objective weights must be finite and nonnegative.")
    if problem.mode == "independent_denominator" and weights.death != 0:
        raise ValueError("Independent denominator mode requires death weight lambda_D=0.")
    if problem.mode == "sim_informed_denominator" and weights.death <= 0:
        raise ValueError("SIM-informed denominator mode requires death weight lambda_D>0.")


_ILR_BASIS_CACHE: dict[int, np.ndarray] = {}


def _ilr_basis(count: int) -> np.ndarray:
    """The (count-1, count) ILR (Helmert-like) basis — depends ONLY on the category count, so it is
    built once and cached, never per loss evaluation."""
    cached = _ILR_BASIS_CACHE.get(count)
    if cached is not None:
        return cached
    basis = np.zeros((max(count - 1, 0), count), dtype=np.float64)
    for j in range(count - 1):
        scale = math.sqrt((j + 1) / (j + 2))
        basis[j, : j + 1] = scale / (j + 1)
        basis[j, j + 1] = -scale
    _ILR_BASIS_CACHE[count] = basis
    return basis


def _ilr_coords(x: np.ndarray, basis: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """ILR coordinates of rows of ``x`` as a single matmul against the cached basis.

    Equivalent to the old per-coordinate Python loop (``coords[j] = sum(log_x * row_j)``) but
    vectorized: ``log_x @ basis.T``. Row-wise, so indexing a subset commutes with the transform."""
    x = np.maximum(x, epsilon)
    x = x / x.sum(axis=-1, keepdims=True)
    log_x = np.log(x)
    return log_x @ basis.T


def _ilr_np(x: np.ndarray, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat wrapper: returns (coords, basis). Prefer :func:`_ilr_coords` with a cached basis."""
    count = x.shape[-1]
    basis = _ilr_basis(count)
    if count <= 1:
        return np.array([]), np.array([])
    return _ilr_coords(x, basis, epsilon), basis


@dataclass
class _LossConstants:
    """Problem-invariant quantities hoisted out of the per-iteration loss hot loop.

    The solver calls :func:`evaluate_population_loss` hundreds of times with a CONSTANT ``problem`` and
    a changing iterate. Everything derived only from ``problem`` — the anchor mask, the survival tensor
    (``1 - death_rate``), the death-term masks, and the race prior's ILR (the *target* never changes) —
    is computed ONCE here and cached on the problem, instead of being rebuilt every evaluation.
    """
    anchors: np.ndarray
    anchor_mask: np.ndarray
    survival: np.ndarray | None
    death_dr: np.ndarray | None
    death_mask: np.ndarray | None
    race_mask: np.ndarray | None       # (S,T,A,X) cells with a fully-observed race prior
    race_basis: np.ndarray | None
    race_prior_masked_ilr: np.ndarray | None  # ILR of the prior for masked cells (target; constant)


def _loss_constants(problem: PopulationTensorProblem) -> _LossConstants:
    cached = getattr(problem, "_cached_loss_constants", None)
    if cached is not None:
        return cached
    n = problem.n_cells
    shape = problem.shape
    anchors = np.asarray(problem.anchors, dtype=np.float64)
    anchor_mask = ~np.isnan(anchors)

    survival = None
    if problem.death_rates is not None:
        dr = np.nan_to_num(np.asarray(problem.death_rates, dtype=np.float64).reshape(shape), nan=0.0)
        survival = 1.0 - dr

    death_dr = death_mask = None
    if problem.sim_deaths is not None:
        death_dr = (
            np.asarray(problem.death_rates, dtype=np.float64)
            if problem.death_rates is not None else np.full(n, np.nan)
        )
        sim_deaths = np.asarray(problem.sim_deaths, dtype=np.float64)
        death_mask = ~np.isnan(sim_deaths) & ~np.isnan(death_dr)

    race_mask = race_basis = race_prior_masked_ilr = None
    if problem.race_composition_prior is not None:
        prior = np.asarray(problem.race_composition_prior, dtype=np.float64).reshape(shape)
        race_mask = ~np.isnan(prior).any(axis=-1)
        race_basis = _ilr_basis(shape[-1])
        race_prior_masked_ilr = _ilr_coords(prior[race_mask], race_basis)

    const = _LossConstants(
        anchors=anchors, anchor_mask=anchor_mask, survival=survival,
        death_dr=death_dr, death_mask=death_mask, race_mask=race_mask,
        race_basis=race_basis, race_prior_masked_ilr=race_prior_masked_ilr,
    )
    object.__setattr__(problem, "_cached_loss_constants", const)
    return const


def evaluate_population_loss(
    problem: PopulationTensorProblem,
    population: tuple[float, ...] | list[float],
    migration: tuple[float, ...] | list[float],
) -> PopulationLossEvaluation:
    validate_population_problem(problem)
    n = problem.n_cells
    if len(population) != n or len(migration) != n:
        raise ValueError("Population and migration vectors must match problem n_cells.")
    
    P = np.array(population, dtype=np.float64)
    M = np.array(migration, dtype=np.float64)
    
    if not np.isfinite(P).all() or not np.isfinite(M).all():
        raise ValueError("Population objective received a non-finite iterate.")

    gp = np.zeros(n, dtype=np.float64)
    gm = np.zeros(n, dtype=np.float64)
    terms = {name: 0.0 for name in ("anchor", "aging", "birth", "death", "migration", "migration_total", "race", "age_smooth")}
    w = problem.weights
    shape = problem.shape
    s_count, t_count, a_count, x_count, r_count = shape
    
    P_tens = P.reshape(shape)
    M_tens = M.reshape(shape)
    gp_tens = gp.reshape(shape)
    gm_tens = gm.reshape(shape)

    # Problem-invariant constants (anchor mask, survival, death/race masks, target race ILR) are
    # computed ONCE and cached on the problem — not rebuilt on every one of the solver's hundreds of
    # loss evaluations. This is the dominant per-iteration cost fix (see _LossConstants).
    const = _loss_constants(problem)

    # 1. Anchors  (problem fields are numpy arrays with NaN sentinels — read directly, no rebuild)
    anchors_np = const.anchors
    anchor_mask = const.anchor_mask
    if w.anchor > 0 and anchor_mask.any():
        residual = P[anchor_mask] - anchors_np[anchor_mask]
        terms["anchor"] = float(w.anchor * np.sum(residual**2))
        gp[anchor_mask] += 2.0 * w.anchor * residual

    # 2. Aging
    if w.aging > 0 and t_count > 1 and a_count > 1:
        # survival = 1 - nan_to_num(death_rates), constant across iterations (cached).
        survival = const.survival if const.survival is not None else np.ones(shape, dtype=np.float64)

        P_curr = P_tens[:, 1:, 1:, :, :]
        P_prior = P_tens[:, :-1, :-1, :, :]
        surv_prior = survival[:, :-1, :-1, :, :]
        M_eta = M_tens[:, :-1, 1:, :, :]
        
        expected = P_prior * surv_prior + M_eta
        
        # Terminal age group
        P_terminal = P_tens[:, :-1, -1:, :, :]
        surv_terminal = survival[:, :-1, -1:, :, :]
        expected[:, :, -1:, :, :] += P_terminal * surv_terminal
        
        residual = P_curr - expected
        terms["aging"] = float(w.aging * np.sum(residual**2))
        
        grad_res = 2.0 * w.aging * residual
        gp_tens[:, 1:, 1:, :, :] += grad_res
        gp_tens[:, :-1, :-1, :, :] -= grad_res * surv_prior
        gp_tens[:, :-1, -1:, :, :] -= grad_res[:, :, -1:, :, :] * surv_terminal
        gm_tens[:, :-1, 1:, :, :] -= grad_res

    # 3. Births
    if w.birth > 0 and problem.births is not None and t_count > 1:
        births = np.asarray(problem.births, dtype=np.float64).reshape((s_count, t_count, x_count, r_count))
        P_curr = P_tens[:, 1:, 0, :, :]
        B_prior = births[:, :-1, :, :]
        M_eta = M_tens[:, :-1, 0, :, :]
        
        mask = ~np.isnan(B_prior)
        if mask.any():
            residual = P_curr[mask] - B_prior[mask] - M_eta[mask]
            terms["birth"] = float(w.birth * np.sum(residual**2))
            
            grad_res = 2.0 * w.birth * residual
            gp_tens[:, 1:, 0, :, :][mask] += grad_res
            gm_tens[:, :-1, 0, :, :][mask] -= grad_res

    # 4. Deaths  (death_rates flat + the ~isnan mask are constant -> cached)
    if w.death > 0 and problem.sim_deaths is not None:
        dr = const.death_dr
        sim_deaths = np.asarray(problem.sim_deaths, dtype=np.float64)
        mask = const.death_mask
        if mask.any():
            residual = dr[mask] * P[mask] - sim_deaths[mask]
            terms["death"] = float(w.death * np.sum(residual**2))
            gp[mask] += 2.0 * w.death * residual * dr[mask]

    # 5. Race  (race_mask, the ILR basis, and the TARGET prior ILR are all constant -> cached; only the
    # observation ILR depends on the iterate. The old code recomputed the target ILR + basis every
    # evaluation, which was ~half of the single dominant cost of the whole solve.)
    if w.race > 0 and const.race_mask is not None:
        mask = const.race_mask
        basis = const.race_basis
        P_masked = P_tens[mask]
        valid = P_masked.sum(axis=-1) > 1e-12
        P_valid = P_masked[valid]

        if len(P_valid) > 0:
            obs_ilr = _ilr_coords(P_valid, basis)
            tgt_ilr = const.race_prior_masked_ilr[valid]  # ilr(prior)[valid] == ilr(prior[valid]) (row-wise)
            residual = obs_ilr - tgt_ilr

            terms["race"] = float(w.race * np.sum(residual**2))

            deriv = residual @ basis
            grad = 2.0 * w.race * deriv / np.maximum(P_valid, 1e-12)

            flat_mask = np.zeros(gp_tens.shape[:-1], dtype=bool)
            flat_mask[mask] = valid
            gp_tens[flat_mask] += grad
            
    # 6. Migration Smooth
    if w.migration > 0 and t_count >= 3:
        i2 = M_tens[:, 2:, :, :, :]
        i1 = M_tens[:, 1:-1, :, :, :]
        i0 = M_tens[:, :-2, :, :, :]
        residual = i2 - 2.0 * i1 + i0
        terms["migration"] = float(w.migration * np.sum(residual**2))
        
        grad_res = 2.0 * w.migration * residual
        gm_tens[:, 2:, :, :, :] += grad_res
        gm_tens[:, 1:-1, :, :, :] -= 2.0 * grad_res
        gm_tens[:, :-2, :, :, :] += grad_res

    # 6b. Migration total (per locality-year net-flow residual anchor, MSD §2.8.7)
    if w.migration_total > 0 and problem.migration_locality_totals is not None:
        mig_obs = np.asarray(problem.migration_locality_totals, dtype=np.float64).reshape((s_count, t_count))
        mask_st = ~np.isnan(mig_obs)
        if mask_st.any():
            # Observed net migration for (s,t) is the sum of eta over (a,x,r).
            flow = M_tens.sum(axis=(2, 3, 4))
            residual_st = np.where(mask_st, flow - mig_obs, 0.0)
            terms["migration_total"] = float(w.migration_total * np.sum(residual_st**2))
            grad_st = 2.0 * w.migration_total * residual_st
            # d(flow_st)/d(eta_{s,t,a,x,r}) = 1 for every cell in the slab.
            gm_tens += grad_st[:, :, None, None, None]

    # 7. Age Smooth
    if w.age_smooth > 0 and a_count >= 3:
        i2 = P_tens[:, :, 2:, :, :]
        i1 = P_tens[:, :, 1:-1, :, :]
        i0 = P_tens[:, :, :-2, :, :]
        residual = i2 - 2.0 * i1 + i0
        terms["age_smooth"] = float(w.age_smooth * np.sum(residual**2))
        
        grad_res = 2.0 * w.age_smooth * residual
        gp_tens[:, :, 2:, :, :] += grad_res
        gp_tens[:, :, 1:-1, :, :] -= 2.0 * grad_res
        gp_tens[:, :, :-2, :, :] += grad_res

    return PopulationLossEvaluation(
        total=float(sum(terms.values())),
        terms=terms,
        population_gradient=gp,  # numpy arrays, returned directly — no per-element Python conversion
        migration_gradient=gm,
    )
