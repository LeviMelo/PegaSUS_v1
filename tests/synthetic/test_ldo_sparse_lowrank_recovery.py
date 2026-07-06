"""LDO sparse+low-rank (CPW) recovery — the S/L identifiability acceptance gate.

The Chandrasekaran–Parrilo–Willsky decomposition ``Ω = S − L`` must separate two
roles that look identical to a naive readout:

  * a **direct edge** (a genuine pairwise conditional dependency) belongs in the
    sparse ``S`` and is reported as a ``contemporaneous``/direct link;
  * a **shared latent driver** (a factor co-moving many variables — an epidemic
    wave, a reporting shock) belongs in the low-rank ``L`` and is reported as a
    ``latent_shared`` (confounded) pair.

The audit found the estimator had *no* fixed ``lambda2`` operating point that kept
both roles: at small ``lambda2`` ``S`` collapsed to diagonal and the direct edge was
misclassified as ``latent_shared``; at large ``lambda2`` ``L`` collapsed to rank 0
and shared drivers became a dense direct-edge clique; and in between, a single direct
edge was *double-reported* as both a direct and a latent_shared link. The fix is the
CPW **incoherence identifiability condition** enforced at readout: a low-rank factor
must be spread across ≥ ``min_factor_support`` variables, else it is a direct edge,
not a driver; and a pair is reported in exactly one role (mutual exclusion).

This test plants a known mix and asserts a clean split, across seeds — the gate that
must stay green for every ``Hypotheses.parquet`` edge classification to be meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest

from pegasus.ldo.lowrank import fit_sparse_plus_lowrank


def _planted_mix(seed: int, *, p: int = 8, factor_vars=(0, 1, 2), direct=(3, 4),
                 n: int = 4000, factor_strength: float = 1.0, direct_coef: float = 0.9):
    """corr matrix from: one shared factor loading `factor_vars`, one direct edge
    `direct`, and independent noise on the rest."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    X = rng.standard_normal((n, p)) * 0.4
    for v in factor_vars:
        X[:, v] += factor_strength * f
    i, j = direct
    X[:, j] += direct_coef * X[:, i]
    return np.corrcoef(X, rowvar=False)


@pytest.mark.parametrize("seed", range(6))
def test_direct_edge_and_shared_factor_are_cleanly_separated(seed: int):
    factor_vars, direct = (0, 1, 2), (3, 4)
    C = _planted_mix(seed, factor_vars=factor_vars, direct=direct)
    fit = fit_sparse_plus_lowrank(C, lambda1=0.1, lambda2=0.1)

    direct_pairs = {tuple(sorted((a, b))) for a, b, _ in fit.direct_edges}
    shared_pairs = {tuple(sorted((a, b))) for a, b, _ in fit.latent_shared}
    want_direct = tuple(sorted(direct))
    want_shared = {tuple(sorted((a, b))) for a in factor_vars for b in factor_vars if a < b}

    # 1. the genuine direct edge is recovered in S (not swallowed by L)
    assert want_direct in direct_pairs, f"seed {seed}: direct edge {want_direct} missing from S; got {sorted(direct_pairs)}"
    # 2. the shared-factor pairs are recovered in L
    assert want_shared.issubset(shared_pairs), f"seed {seed}: factor pairs {want_shared} missing from L; got {sorted(shared_pairs)}"
    # 3. the direct edge is NOT double-reported as latent_shared (the incoherence gate)
    assert want_direct not in shared_pairs, f"seed {seed}: direct edge {want_direct} double-reported as latent_shared"
    # 4. mutual exclusion: no pair is in both roles
    assert direct_pairs.isdisjoint(shared_pairs), f"seed {seed}: overlap {direct_pairs & shared_pairs}"


def test_two_variable_factor_is_not_reported_as_a_driver():
    """A rank-1 component concentrated on exactly 2 variables is a direct edge, not a
    shared driver — the incoherence condition. With no genuine multi-variable factor,
    latent_shared must be empty."""
    rng = np.random.default_rng(7)
    n, p = 4000, 6
    X = rng.standard_normal((n, p)) * 0.5
    X[:, 1] += 0.9 * X[:, 0]  # a single strong pairwise edge, no spread driver
    C = np.corrcoef(X, rowvar=False)
    fit = fit_sparse_plus_lowrank(C, lambda1=0.1, lambda2=0.1)
    shared_pairs = {tuple(sorted((a, b))) for a, b, _ in fit.latent_shared}
    assert shared_pairs == set(), f"a 2-variable component leaked into latent_shared: {sorted(shared_pairs)}"
    assert tuple(sorted((0, 1))) in {tuple(sorted((a, b))) for a, b, _ in fit.direct_edges}
