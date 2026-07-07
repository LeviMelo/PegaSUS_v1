"""WP8 / §V.1 — mixed-precision (float32-bulk) LVGLASSO with float64 reductions + escalation.

The national compute envelope wants the ADMM bulk iterates in float32 (half the working set),
but the log-det/eigh *reductions* must stay float64 or small eigenvalues corrupt. And an
ill-conditioned covariance must ESCALATE float32→float64 (§V.4), not silently degrade. The
envelope byte model must also be honest: a float64 fit is 8 B/cell, not the old flat 4 B.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.envelope import estimate_ldo_bytes
from pegasus.ldo.lowrank import fit_sparse_plus_lowrank


def _well_conditioned_cov(seed: int = 0, p: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # a sparse-plus-lowrank precision → invert to a well-conditioned covariance
    A = np.eye(p)
    A[0, 1] = A[1, 0] = 0.4
    A[2, 3] = A[3, 2] = -0.35
    L = np.zeros((p, p))
    v = rng.standard_normal(p)
    L += 0.2 * np.outer(v, v) / (v @ v)
    prec = A + 0.3 * np.eye(p)  # keep PD, moderate conditioning
    cov = np.linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def test_float32_bulk_matches_float64_on_well_conditioned_cov():
    C = _well_conditioned_cov()
    f64 = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.05, work_dtype=np.float64)
    f32 = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.05, work_dtype=np.float32)
    assert f64.work_dtype == "float64" and not f64.cond_escalated
    assert f32.work_dtype == "float32" and not f32.cond_escalated  # well-conditioned → no escalation
    # The float32 bulk recovers the same precision structure to a tight tolerance (the eigh
    # reductions ran in float64 in both), and the returned arrays are float64 for readout.
    assert f32.precision.dtype == np.float64
    assert np.allclose(f32.precision, f64.precision, atol=1e-3, rtol=1e-3)
    # Same selected direct edges.
    assert {(i, j) for i, j, _ in f32.direct_edges} == {(i, j) for i, j, _ in f64.direct_edges}


def test_float64_default_is_byte_identical():
    C = _well_conditioned_cov(seed=3)
    a = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.05)                     # default
    b = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.05, work_dtype=np.float64)
    assert np.array_equal(a.precision, b.precision)
    assert a.iterations == b.iterations and a.converged == b.converged


def test_ill_conditioned_cov_escalates_to_float64():
    # A near-singular correlation (one direction with tiny variance) → huge cond(C).
    p = 8
    C = np.eye(p)
    C[0, 1] = C[1, 0] = 0.999999   # cond ~ 2e6+ before the +1e-4 ridge; push past the threshold
    fit = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.05,
                                  work_dtype=np.float32, cond_escalate_threshold=1e3)
    assert fit.cond_escalated is True
    assert fit.work_dtype == "float64"


def test_envelope_byte_model_is_honest():
    # float64 bulk must not be under-counted: the two bulk terms (samples, cov) are 8 B, so
    # the float64 estimate is strictly larger than the float32-bulk estimate (never equal, as
    # the old flat-4B model made them).
    kw = dict(p=40, S=5000, T=25, K=8)
    f64 = estimate_ldo_bytes(**kw, float32_bulk=False)
    f32 = estimate_ldo_bytes(**kw, float32_bulk=True)
    assert f64 > f32
    # The bulk terms differ by exactly a factor of two (spatial float64 term is shared).
    pk = kw["p"] * (kw["K"] + 1)
    spatial = 3 * kw["S"] * kw["S"] * 8
    assert (f64 - spatial) == 2 * (f32 - spatial)
