"""LDO randomized low-rank factor extraction (§V.3) — exact certifies approximate.

The shared-driver factors are few, so at scale the low-rank component L's eigenfactors
are recovered by randomized SVD (Halko–Martinsson–Tropp) in O(p²·r) instead of a dense
O(p³) eigh. Because it is a post-convergence readout on a genuinely low-rank L (above a
noise floor), the randomized and exact readouts must agree (§V.6). This pins that.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.lowrank import fit_sparse_plus_lowrank


def _planted_corr(seed: int, p: int = 60):
    rng = np.random.default_rng(seed)
    n = 3000
    f = rng.standard_normal(n)
    X = rng.standard_normal((n, p)) * 0.5
    for v in (0, 1, 2):        # a 3-variable shared factor
        X[:, v] += f
    X[:, 4] += 0.9 * X[:, 3]   # a direct edge
    return np.corrcoef(X, rowvar=False)


def _pairs(fit):
    ls = {tuple(sorted((a, b))) for a, b, _ in fit.latent_shared}
    de = {tuple(sorted((a, b))) for a, b, _ in fit.direct_edges}
    return ls, de


def test_randomized_factor_readout_agrees_with_exact():
    for seed in range(4):
        C = _planted_corr(seed)
        exact = fit_sparse_plus_lowrank(C, lambda1=0.1, lambda2=0.1, randomized_factors=False)
        rand = fit_sparse_plus_lowrank(C, lambda1=0.1, lambda2=0.1, randomized_factors=True,
                                       factor_rank_cap=12, seed=0)
        els, eds = _pairs(exact)
        rls, rds = _pairs(rand)
        assert els == rls, f"seed {seed}: latent_shared differs exact={sorted(els)} rand={sorted(rls)}"
        assert eds == rds, f"seed {seed}: direct_edges differ exact={sorted(eds)} rand={sorted(rds)}"
        # and the recovery is clean: the true factor pairs and only them
        assert els == {(0, 1), (0, 2), (1, 2)}, f"seed {seed}: spurious/absent factor pairs {sorted(els)}"
        assert (3, 4) in eds
