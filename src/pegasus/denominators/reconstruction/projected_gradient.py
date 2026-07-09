from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pegasus.denominators.reconstruction.loss import evaluate_population_loss, validate_population_problem
from pegasus.denominators.reconstruction.schema import PopulationSolverTelemetry, PopulationTensorProblem


@dataclass(frozen=True)
class PopulationOptimizationResult:
    # numpy arrays, not Python float tuples — see PopulationLossEvaluation; the solver result at
    # national scale is ~200M cells, and tuple(float(x) for x in ...) was ~1 GB + a redundant
    # numpy→list→numpy round-trip at the emission site.
    population: np.ndarray
    migration: np.ndarray
    telemetry: PopulationSolverTelemetry


def _simplex_projection(values: list[float], total: float) -> list[float]:
    if total < 0:
        raise ValueError("Closure totals must be nonnegative.")
    if not values:
        if abs(total) > 1e-9:
            raise ValueError("Closure total cannot be satisfied without free cells.")
        return []
    if total <= 1e-12:
        return [0.0 for _ in values]
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
    # Fields are numpy arrays: NaN is the 'absent' sentinel (was None), so element checks use isnan/isfinite.
    hard = problem.hard_anchor_mask if problem.hard_anchor_mask is not None else np.zeros(problem.n_cells, dtype=bool)
    for i, locked in enumerate(hard):
        if locked:
            anchor = float(problem.anchors[i])
            if not math.isfinite(anchor) or anchor < 0:
                raise ValueError("Hard population anchors must be present and nonnegative.")
            projected[i] = anchor

    if problem.closure_totals is None:
        return projected
    s_count, t_count, a_count, x_count, r_count = problem.shape
    group_size = a_count * x_count * r_count
    for s in range(s_count):
        for t in range(t_count):
            total = float(problem.closure_totals[s * t_count + t])
            if math.isnan(total):
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
        mb = np.asarray(problem.migration_bounds, dtype=np.float64)
        if bool(np.any((mb < 0) | ~np.isfinite(mb))):
            raise ValueError("Migration bounds must be finite and nonnegative.")
        return mb
    fallback = float(np.nan_to_num(np.asarray(problem.anchors, dtype=np.float64), nan=0.0).max(initial=0.0))
    if problem.closure_totals is not None:
        fallback = max(fallback, float(np.nan_to_num(np.asarray(problem.closure_totals, dtype=np.float64), nan=0.0).max(initial=0.0)))
    return np.full(problem.n_cells, max(fallback, 1.0), dtype=np.float64)


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
                    total = float(problem.migration_totals[total_index])
                    if math.isnan(total):
                        continue
                    indices = [((((s * t_count) + t) * a_count + a) * x_count + x) * r_count + r for s in range(s_count)]
                    group = _bounded_sum_projection([projected[i] for i in indices], [bounds[i] for i in indices], total)
                    for i, value in zip(indices, group, strict=True):
                        projected[i] = value
    return projected


def _initial_population(problem: PopulationTensorProblem) -> list[float]:
    """Warm start. A supplied ``initial_population`` (the closed-form prior-mean tensor from
    ``sidra.population_cube.build.interpolate_census_composition``, or a caller's own iterate)
    wins; otherwise fall back to the observed anchors (0 where unobserved). The projection
    then distributes any closure total the warm start does not already account for."""
    def _finish(vals: np.ndarray) -> list[float]:
        if _fast_projection_supported(problem):
            return [float(v) for v in _np_project_population(problem, vals)]
        return _project_population(problem, [float(v) for v in vals])

    if problem.initial_population is not None:
        return _finish(np.asarray(problem.initial_population, dtype=np.float64))
    anchors = np.nan_to_num(np.asarray(problem.anchors, dtype=np.float64), nan=0.0)
    return _finish(anchors)


def _fast_projection_supported(problem: PopulationTensorProblem) -> bool:
    """The vectorized projections below handle the common denominator case:
    no per-cell hard anchors and no per-stratum migration-total equality (both use
    the additional per-cell/per-group bookkeeping the pure-Python paths implement).
    Everything in the current SIDRA population build hits this path."""
    if problem.hard_anchor_mask is not None and bool(np.asarray(problem.hard_anchor_mask).any()):
        return False
    if problem.migration_totals is not None:
        return False
    return True


def _simplex_project_rows(rows: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row onto {y>=0, sum(y)=total} (Held et al. / Duchi).

    Vectorized equivalent of ``_simplex_projection`` applied per (locality, year)
    closure group -- the hot loop of the dense solver at national/state scale."""
    out = np.zeros_like(rows)
    positive = totals > 1e-12
    if not positive.any():
        return out
    sub = rows[positive]
    tot = totals[positive]
    ordered = np.sort(sub, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    j = np.arange(1, sub.shape[1] + 1)
    condition = ordered - (cumulative - tot[:, None]) / j > 0
    rho = condition.sum(axis=1)
    rho_idx = np.clip(rho - 1, 0, sub.shape[1] - 1)
    css_rho = cumulative[np.arange(sub.shape[0]), rho_idx]
    theta = (css_rho - tot) / np.maximum(rho, 1)
    out[positive] = np.maximum(sub - theta[:, None], 0.0)
    return out


def _np_project_population(problem: PopulationTensorProblem, values: np.ndarray) -> np.ndarray:
    projected = np.maximum(values, 0.0)
    if problem.closure_totals is None:
        return projected
    s_count, t_count, a_count, x_count, r_count = problem.shape
    group_size = a_count * x_count * r_count
    grouped = projected.reshape(s_count * t_count, group_size)
    totals = np.asarray(problem.closure_totals, dtype=np.float64)
    mask = ~np.isnan(totals)
    if mask.any():
        grouped[mask] = _simplex_project_rows(grouped[mask], totals[mask])
    return grouped.reshape(-1)


def _np_project_migration(problem: PopulationTensorProblem, values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(values, -bounds, bounds)


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
    if _fast_projection_supported(problem):
        # POP-02 GPU: solve on the GPU (torch SPG, float32) when population_solver is CUDA-enabled and the
        # block is large enough to beat host<->device transfer. Same algorithm; recovery validated to
        # ~3e-7/cell in f32 (loss/gradient are machine-identical via autograd). Any CUDA error (e.g. VRAM
        # pressure) falls back to the CPU path -- a device reason never fails a run.
        from pegasus.denominators.reconstruction.torch_solver import (
            maybe_population_gpu_plan,
            solve_population_tensor_torch,
        )

        _plan = maybe_population_gpu_plan(problem)
        if _plan is not None:
            try:
                return solve_population_tensor_torch(
                    problem, plan=_plan, max_iterations=max_iterations, tolerance=tolerance
                )
            except Exception as _exc:  # noqa: BLE001 -- graceful device fallback, never fail a run
                import warnings

                warnings.warn(
                    f"population GPU solve fell back to CPU ({type(_exc).__name__}: {_exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return _solve_projected_gradient_vectorized(
            problem, max_iterations=max_iterations, tolerance=tolerance, initial_step_size=initial_step_size,
        )

    population = _initial_population(problem)
    bounds = _migration_bounds(problem)
    _init_mig = problem.initial_migration if problem.initial_migration is not None else np.zeros(problem.n_cells)
    migration = _project_migration(problem, list(_init_mig), bounds)
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
    return PopulationOptimizationResult(np.asarray(population, dtype=np.float64), np.asarray(migration, dtype=np.float64), telemetry)


def _solve_projected_gradient_vectorized(
    problem: PopulationTensorProblem,
    *,
    max_iterations: int,
    tolerance: float,
    initial_step_size: float,
) -> PopulationOptimizationResult:
    """Numpy projected-gradient with Barzilai-Borwein step + Armijo backtracking.

    Two fixes over the naive fixed-step loop, both required for a real (state/national)
    denominator tensor: (1) the projections and vector updates run in numpy, not
    Python loops over ~1e6 cells (per-iteration cost drops from tens of seconds to
    ~10ms); (2) the step is scaled to the local curvature (BB) with a scale-correct
    Armijo test, so a gradient of magnitude ~1e6 at a ~1e10 objective takes real steps
    instead of overshooting once then stalling (the old loop terminated after ~2
    iterations, leaving intercensal cells at their uniform seed -> uniform race/age).
    """
    n = problem.n_cells
    population = np.array(_initial_population(problem), dtype=np.float64)
    bounds = np.asarray(_migration_bounds(problem), dtype=np.float64)
    init_migration = (
        np.asarray(problem.initial_migration, dtype=np.float64)
        if problem.initial_migration is not None else np.zeros(n, dtype=np.float64)
    )
    migration = _np_project_migration(problem, init_migration, bounds)

    evaluation = evaluate_population_loss(problem, population, migration)
    grad_p = np.asarray(evaluation.population_gradient, dtype=np.float64)
    grad_m = np.asarray(evaluation.migration_gradient, dtype=np.float64)
    initial_objective = evaluation.total
    previous_objective = initial_objective
    # First-step scale (standard SPG init): 1/||g||_inf so the very first projected step
    # P(x - alpha*g) moves ~1 unit per cell rather than saturating the simplex with a
    # fixed alpha=1 against a ~1e6 gradient (which makes the projected direction a giant
    # jump and the line search stall). Barzilai-Borwein takes over from iteration 2.
    grad_inf = max(float(np.abs(grad_p).max(initial=0.0)), float(np.abs(grad_m).max(initial=0.0)), 1.0)
    step_size = min(initial_step_size, 1.0 / grad_inf)
    relative_change = 0.0
    projected_norm = math.inf
    converged = False
    iterations = 0

    prev_population: np.ndarray | None = None
    prev_migration: np.ndarray | None = None
    prev_grad_p: np.ndarray | None = None
    prev_grad_m: np.ndarray | None = None
    stall = 0
    small_change = 0
    # Non-monotone reference window (GLL): the Spectral Projected Gradient method
    # (Birgin-Martinez-Raydan) pairs the BB step with a line search that accepts a
    # trial against the MAX objective over the last M steps, not the current one.
    # BB steps are intentionally non-monotone, so a strict monotone Armijo test rejects
    # them and backtracks ~50x per iteration (hundreds of loss evals); the non-monotone
    # window lets the BB step through on the first try, cutting the solve ~20x.
    memory_window = 10
    objective_history = [initial_objective]

    for iteration in range(1, max_iterations + 1):
        # Projected-gradient stationarity (unit step): ||x - proj(x - g)||.
        stat_p = population - _np_project_population(problem, population - grad_p)
        stat_m = migration - _np_project_migration(problem, migration - grad_m, bounds)
        projected_norm = float(math.sqrt(float(stat_p @ stat_p) + float(stat_m @ stat_m)))
        if projected_norm <= tolerance:
            converged = True
            iterations = iteration - 1
            break

        # Barzilai-Borwein step alpha = <s,s>/<s,y> from the last accepted move.
        if prev_population is not None:
            s_p = population - prev_population
            s_m = migration - prev_migration
            y_p = grad_p - prev_grad_p
            y_m = grad_m - prev_grad_m
            sy = float(s_p @ y_p) + float(s_m @ y_m)
            ss = float(s_p @ s_p) + float(s_m @ s_m)
            if sy > 1e-30 and ss > 0.0:
                step_size = min(max(ss / sy, 1e-12), 1e12)

        # SPG line search: backtrack along the projected BB direction
        # d = P(x - alpha*g) - x using the directional derivative <g,d> (<= 0), NOT
        # ||g||^2 (whose ~1e18 magnitude at a 1e10 objective would force ~14 spurious
        # backtracks). Trials x + lam*d are convex combinations of two feasible points,
        # so they stay feasible with NO re-projection inside the loop -- the BB step is
        # accepted in ~1 evaluation, turning tens-of-seconds iterations into ~1s.
        reference = max(objective_history)
        proj_p = _np_project_population(problem, population - step_size * grad_p)
        proj_m = _np_project_migration(problem, migration - step_size * grad_m, bounds)
        dir_p = proj_p - population
        dir_m = proj_m - migration
        directional = float(grad_p @ dir_p) + float(grad_m @ dir_m)
        if directional >= -1e-30:
            # BB step yielded a non-descent projected direction (rare): retry once with a
            # small safeguard step, which is guaranteed descent for a projected gradient.
            step_size = max(min(step_size, 1.0) * 1e-3, 1e-14)
            proj_p = _np_project_population(problem, population - step_size * grad_p)
            proj_m = _np_project_migration(problem, migration - step_size * grad_m, bounds)
            dir_p = proj_p - population
            dir_m = proj_m - migration
            directional = float(grad_p @ dir_p) + float(grad_m @ dir_m)
        lam = 1.0
        accepted = False
        trial_population = population
        trial_migration = migration
        trial = evaluation
        f0 = evaluation.total
        # Safeguarded quadratic-interpolation backtracking. Blind halving needed ~18-22 evals per
        # iteration on the real (warm-started, near-optimal, underdetermined) tensor -- the census
        # warm start is already the optimum, so no line search finds descent and the loop just burns
        # evals. Two fixes: (a) when a descent step DOES exist, fit the 1-D quadratic through
        # (0,f0,slope=directional) and (lam,trial) and jump to its minimizer (2-3 evals, not ~18);
        # (b) cap the doomed search at 8 backtracks instead of 30. Combined with the faster stall
        # exit below, the underdetermined case drops from ~240 wasted evals to ~20.
        for _ in range(8):
            trial_population = population + lam * dir_p
            trial_migration = migration + lam * dir_m
            trial = evaluate_population_loss(problem, trial_population, trial_migration)
            if trial.total <= reference + 1e-4 * lam * directional:
                accepted = True
                break
            denom = 2.0 * (trial.total - f0 - directional * lam)
            lam_quad = (-directional * lam * lam / denom) if denom > 1e-30 else 0.5 * lam
            lam = min(max(lam_quad, 0.1 * lam), 0.5 * lam)  # safeguard to a real fraction of lam
        if not accepted:
            # No feasible descent along the projected gradient: the current point is a constrained
            # (possibly underdetermined) stationary point -- the census warm start IS the answer
            # (build.py §2.8.10 note). Bail after a few stalls instead of grinding the full budget;
            # a well-conditioned problem accepts real steps (stall resets to 0) and never gets here.
            stall += 1
            step_size = max(step_size * 0.1, 1e-14)
            iterations = iteration
            if stall >= 3:
                break
            continue
        stall = 0

        prev_population, prev_migration = population, migration
        prev_grad_p, prev_grad_m = grad_p, grad_m
        population, migration = trial_population, trial_migration
        evaluation = trial
        grad_p = np.array(evaluation.population_gradient, dtype=np.float64)
        grad_m = np.array(evaluation.migration_gradient, dtype=np.float64)
        objective_history.append(evaluation.total)
        if len(objective_history) > memory_window:
            objective_history.pop(0)
        relative_change = abs(previous_objective - evaluation.total) / max(abs(previous_objective), 1.0)
        previous_objective = evaluation.total
        iterations = iteration
        # Progress-stall convergence: BB descent that has flattened for several
        # consecutive steps is at a numerical optimum (projected_norm may never reach a
        # tight absolute tolerance for a 1e6-cell tensor with population-scale values).
        small_change = small_change + 1 if relative_change <= tolerance else 0
        if small_change >= 5:
            converged = True
            break

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
    return PopulationOptimizationResult(np.asarray(population, dtype=np.float64), np.asarray(migration, dtype=np.float64), telemetry)
