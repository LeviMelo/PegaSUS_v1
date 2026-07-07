"""LDO CPW identifiability guard + adaptive-rho ADMM integrity (W5 + W6).

The sparse+low-rank split ``Ω = S − L`` is only identifiable when S is *spiky* and L is
*spread* (Chandrasekaran–Parrilo–Willsky incoherence). When the sparse mass collapses into
L's span the split is arbitrary — an edge read off S may be shared latent structure. The fit
now scores this (``incoherence`` / ``well_identified``) WITHOUT altering S/L, and the ADMM
uses Boyd residual balancing (adaptive rho) that reaches the same fixed point faster; the
fixed-rho path stays byte-identical.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.lowrank import fit_sparse_plus_lowrank


def _identifiable_cov(seed: int = 0, *, p: int = 12, n: int = 6000) -> np.ndarray:
    """A genuine spread low-rank driver (loads half the vars) + a couple of sparse edges."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    X = rng.standard_normal((n, p)) * 0.5
    for v in range(p // 2):          # a driver spread across many variables → spread L
        X[:, v] += 0.9 * f
    X[:, p - 1] += 0.8 * X[:, p - 2]  # one isolated sparse edge → spiky S
    return np.corrcoef(X, rowvar=False)


def _degenerate_cov(seed: int = 1, *, p: int = 12, n: int = 6000) -> np.ndarray:
    """Near-degenerate: the 'driver' concentrates on 3 vars and one carries a strong direct
    edge, so the sparse mass sits inside the low-rank span — the split is not identifiable."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    X = rng.standard_normal((n, p)) * 0.5
    concentrated = (0, 1, 2)
    for v in concentrated:
        X[:, v] += 1.6 * f          # a coherent (spiky) low-rank component on few vars
    X[:, 1] += 1.4 * X[:, 0]        # sparse edge living on the SAME concentrated coordinates
    return np.corrcoef(X, rowvar=False)


def test_identifiability_score_flags_the_degenerate_split():
    good = fit_sparse_plus_lowrank(_identifiable_cov(), lambda1=0.08, lambda2=0.05)
    bad = fit_sparse_plus_lowrank(_degenerate_cov(), lambda1=0.08, lambda2=0.05)

    assert good.incoherence is not None and bad.incoherence is not None
    assert 0.0 <= good.incoherence <= 1.0 and 0.0 <= bad.incoherence <= 1.0
    # A spread driver + spiky S is well identified; a concentrated split is not.
    assert good.well_identified is True, f"spread split flagged unidentified (score={good.incoherence:.3f})"
    assert bad.well_identified is False, f"concentrated split passed the guard (score={bad.incoherence:.3f})"
    assert good.incoherence > bad.incoherence
    # The diagnostic must NOT change the estimate: same S with the guard on vs a raised threshold.
    ref = fit_sparse_plus_lowrank(_degenerate_cov(), lambda1=0.08, lambda2=0.05,
                                  incoherence_threshold=0.0)
    assert np.array_equal(bad.S, ref.S) and np.array_equal(bad.L, ref.L)


def test_adaptive_rho_is_faster_and_default_off_is_byte_identical():
    C = _identifiable_cov(seed=3)
    # A badly-scaled rho makes the fixed-rho residuals lopsided (small primal, large dual) → the
    # relative-primal stopping rule trips far from the true fixed point after many iterations.
    kw = dict(lambda1=0.08, lambda2=0.05, rho=8.0, tol=1e-8, max_iter=50000)

    # Ground truth: the rho-independent fixed point, driven to a much tighter tolerance.
    truth = fit_sparse_plus_lowrank(C, adaptive_rho=False, lambda1=0.08, lambda2=0.05,
                                    rho=8.0, tol=1e-11, max_iter=50000)

    fixed = fit_sparse_plus_lowrank(C, adaptive_rho=False, **kw)
    adapt = fit_sparse_plus_lowrank(C, adaptive_rho=True, **kw)

    assert fixed.converged and adapt.converged
    # Boyd residual balancing reaches the fixed point in far FEWER iterations...
    assert adapt.iterations < fixed.iterations
    # ...and lands ON the true fixed point within 1e-6 (both residuals balanced small), whereas
    # the lopsided fixed-rho stop is still visibly short of it.
    assert np.allclose(adapt.S, truth.S, atol=1e-6)
    assert np.allclose(adapt.L, truth.L, atol=1e-6)
    assert not np.allclose(fixed.S, truth.S, atol=1e-6)  # fixed-rho stopped short → adaptive earns it

    # The DEFAULT path is the exact fixed-rho ADMM (adaptive is opt-in): a plain call must be
    # byte-identical to an explicit adaptive_rho=False — the Boyd branch never fires by default.
    default = fit_sparse_plus_lowrank(C, **kw)
    assert np.array_equal(default.S, fixed.S) and np.array_equal(default.L, fixed.L)
    assert default.iterations == fixed.iterations and default.converged == fixed.converged
