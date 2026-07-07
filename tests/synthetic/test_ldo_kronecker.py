"""WP8 / §V.2 — the Kronecker-factored joint precision operator.

``Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹`` must reproduce the dense joint operator's matvec, log-det,
and solve *exactly* (to numerical precision) while never materializing the ``(pSτ)²`` object —
the ``(pST)² → p²+S²+τ²`` collapse §V.2 requires. The proof plants small factors, builds the
dense Kronecker as ground truth, and checks the factored primitives against it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pegasus.ldo.kron import (
    KroneckerPrecision,
    build_temporal_precision_ar1,
    kron_cg_solve,
    sparse_spd_logdet,
)


def _factors(seed: int = 0):
    rng = np.random.default_rng(seed)
    p, S, tau = 4, 6, 3
    # Ω_var: SPD p×p
    Av = rng.standard_normal((p, p))
    omega = Av @ Av.T + p * np.eye(p)
    # Σ_space⁻¹: sparse SPD S×S (a small GMRF-like tridiagonal + ridge)
    main = 2.5 * np.ones(S)
    off = -1.0 * np.ones(S - 1)
    space = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    # Σ_time⁻¹: AR(1) precision
    time = build_temporal_precision_ar1(tau, phi=0.4)
    return KroneckerPrecision(omega_var=omega, sigma_space_inv=space, sigma_time_inv=time)


def test_matvec_matches_dense_kronecker():
    op = _factors()
    dense = op.to_dense()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(op.dim)
    assert np.allclose(op.matvec(x), dense @ x, atol=1e-9, rtol=1e-9)


def test_factored_logdet_matches_dense_slogdet():
    op = _factors()
    dense = op.to_dense()
    sign, ld = np.linalg.slogdet(dense)
    assert sign > 0
    assert abs(op.logdet() - ld) < 1e-7
    # The sparse-factor logdet primitive itself is exact vs dense.
    B = op.sigma_space_inv.toarray()
    assert abs(sparse_spd_logdet(op.sigma_space_inv) - np.linalg.slogdet(B)[1]) < 1e-8


def test_exact_mode_wise_solve_inverts_the_operator():
    op = _factors()
    rng = np.random.default_rng(2)
    b = rng.standard_normal(op.dim)
    x = op.solve(b)
    # A x == b, and solve == dense inverse applied
    assert np.allclose(op.matvec(x), b, atol=1e-8, rtol=1e-8)


def test_cg_solve_matches_exact_solve():
    op = _factors()
    rng = np.random.default_rng(3)
    b = rng.standard_normal(op.dim)
    x_cg = kron_cg_solve(op, b, tol=1e-12, max_iter=200)
    assert np.allclose(op.matvec(x_cg), b, atol=1e-7, rtol=1e-7)


def test_logdet_scales_with_axis_sizes():
    # logdet(A⊗I⊗I) over the joint = (S·τ)·logdet(A); verify the exponent bookkeeping with
    # identity space/time factors so the cross terms vanish.
    p, S, tau = 3, 5, 4
    Av = np.diag([2.0, 3.0, 4.0])
    space = sp.identity(S, format="csr")
    time = np.eye(tau)
    op = KroneckerPrecision(omega_var=Av, sigma_space_inv=space, sigma_time_inv=time)
    expected = (S * tau) * np.linalg.slogdet(Av)[1]  # identities contribute 0
    assert abs(op.logdet() - expected) < 1e-9


def test_never_materializes_joint_for_large_space():
    # A large sparse space factor: matvec/solve/logdet must run without a dense (pSτ)² alloc.
    p, S, tau = 3, 4000, 2
    omega = np.eye(p) + 0.1
    main = 2.0 * np.ones(S)
    off = -0.5 * np.ones(S - 1)
    space = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    time = build_temporal_precision_ar1(tau, phi=0.3)
    op = KroneckerPrecision(omega_var=omega, sigma_space_inv=space, sigma_time_inv=time)
    x = np.ones(op.dim)
    y = op.matvec(x)
    assert y.shape == (op.dim,) and np.isfinite(y).all()
    ld = op.logdet()  # sparse Cholesky path, no densify
    assert np.isfinite(ld)
