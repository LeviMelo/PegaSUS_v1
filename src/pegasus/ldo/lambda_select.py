"""StARS λ₁ selection for the LDO sparse precision (Liu–Roeder–Wasserman).

Pick the smallest ℓ1 penalty (densest graph) whose subsample selection instability
stays under ``beta`` — the largest stable graph — instead of a hardcoded λ₁.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pegasus.ldo.edges import _subsample_edges
from pegasus.ldo.margins import GaussianField


@dataclass
class LambdaStARSReport:
    lambda_grid: tuple[float, ...]
    instability: tuple[float, ...]      # monotonized ξ(λ), aligned to lambda_grid
    lambda_star: float
    beta: float
    n_effective_subsamples: tuple[int, ...]


def _edge_instability(edge_sets: list[set], runs: int) -> float:
    """StARS total instability ξ = mean_e 2 f_e (1-f_e) over the union of edges seen."""
    if runs < 2:
        return 0.0
    counts: dict[Any, int] = {}
    for s in edge_sets:
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    if not counts:
        return 0.0
    f = np.array(list(counts.values()), dtype=np.float64) / runs
    return float(np.mean(2.0 * f * (1.0 - f)))


def select_lambda_stars(
    field: GaussianField,
    lambda_grid,
    *,
    K: int = 3,
    n_subsamples: int = 12,
    subsample_frac: float = 0.7,
    seed: int = 0,
    beta: float = 0.05,
    fit_kwargs: dict | None = None,
) -> tuple[float, LambdaStARSReport]:
    """Return ``(lambda_star, report)``. Instability is monotonized (running-max over
    the density-ordered grid) so ξ is non-decreasing as λ shrinks; λ* is the smallest
    λ with ξ ≤ beta (fallback: the λ with minimum ξ)."""
    grid = [float(x) for x in lambda_grid]
    if not grid:
        raise ValueError("lambda_grid must be non-empty")
    base_kwargs = dict(fit_kwargs or {})
    _, S, _ = field.shape
    n_keep = max(2, int(round(subsample_frac * S)))

    # Fixed subsample index sets across the whole grid so ξ(λ) is comparable λ-to-λ
    # (only the penalty changes, not which cells were held out).
    rng = np.random.default_rng(seed)
    index_sets = [np.sort(rng.choice(S, size=min(n_keep, S), replace=False)) for _ in range(n_subsamples)]

    raw_xi: list[float] = []
    n_eff: list[int] = []
    for lam in grid:
        kw = dict(base_kwargs)
        kw["lambda1"] = lam
        edge_sets = [_subsample_edges(field, idx, K=K, fit_kwargs=kw) for idx in index_sets]
        edge_sets = [s for s in edge_sets if s is not None]
        n_eff.append(len(edge_sets))
        raw_xi.append(_edge_instability(edge_sets, len(edge_sets)))

    # Density increases as λ decreases: order by descending λ, take the running max so
    # instability is monotone non-decreasing in density (the StARS monotonization).
    order = sorted(range(len(grid)), key=lambda k: grid[k], reverse=True)
    mono = [0.0] * len(grid)
    run = 0.0
    for k in order:
        run = max(run, raw_xi[k])
        mono[k] = run

    stable = [k for k in order if mono[k] <= beta]
    if stable:
        star_k = min(stable, key=lambda k: grid[k])  # smallest λ (densest) that is stable
    else:
        star_k = min(range(len(grid)), key=lambda k: mono[k])

    report = LambdaStARSReport(
        lambda_grid=tuple(grid),
        instability=tuple(mono),
        lambda_star=grid[star_k],
        beta=beta,
        n_effective_subsamples=tuple(n_eff),
    )
    return grid[star_k], report


__all__ = ["select_lambda_stars", "LambdaStARSReport"]
