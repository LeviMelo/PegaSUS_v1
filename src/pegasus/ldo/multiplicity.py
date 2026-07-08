"""Dependence-aware multiple-testing for LDO edges (Theme-5).

Per-edge two-sided partial-correlation p-values at the dependence-corrected effective-n
(``n_eff`` already deflated for spatial autocorrelation/reliability), controlled across the
discovered edges by Benjamini-Hochberg. Running BH on effective-n p-values is what makes the
count honest under dependence.
"""

from __future__ import annotations

import math


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_z_pvalue(partial_corr: float, n_eff: float, n_conditioning: int = 0) -> float:
    """Two-sided Fisher-z p-value for a partial correlation controlling for ``n_conditioning``
    covariates. dof = n_eff − n_conditioning − 3; when dof < 1 the partial correlation is not
    identifiable at this effective sample size, so return 1.0 (uninformative) rather than a
    falsely-confident tiny p. ``n_conditioning=0`` is the zero-order (marginal) case."""
    dof = n_eff - n_conditioning - 3.0
    if dof < 1.0:
        return 1.0
    r = max(min(partial_corr, 0.999), -0.999)
    z = math.atanh(r)
    se = 1.0 / math.sqrt(dof)
    return 2.0 * (1.0 - _phi(abs(z) / se))


def benjamini_hochberg(pvalues: list[float], q: float) -> tuple[list[bool], list[float]]:
    m = len(pvalues)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    qvals = [0.0] * m
    running = 1.0
    # step-up: enforce monotone BH q from the largest p down
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * m / rank)
        qvals[i] = running
    rejected = [qvals[i] <= q for i in range(m)]
    return rejected, qvals


def block_permutation_qhook(pvalues, q, permute):  # optional refinement hook (unused by default)
    return benjamini_hochberg(pvalues, q)


__all__ = ["fisher_z_pvalue", "benjamini_hochberg", "block_permutation_qhook"]
