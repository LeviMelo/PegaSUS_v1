"""Stochastic log-determinant via Hutchinson + stochastic Lanczos quadrature (MSD-III §V.3).

The Gaussian likelihood needs ``log det Ω`` for an ``O(n)``-cell precision. The exact
``O(n³)`` Cholesky/eigh does not scale; §V.3 prescribes an unbiased **matrix-free** estimator
whose variance is controlled by the probe count (as in GPyTorch): ``log det A = tr(log A) ≈
E_z[zᵀ log(A) z]`` with Rademacher ``z``, each quadratic form estimated by ``m`` Lanczos steps
(Ubaru–Chen–Saad). It touches ``A`` only through matvecs, so a sparse/Kronecker operator never
densifies, and is an anytime evaluator (more probes ⇒ tighter) feeding the §V.5 controller and
the §V.6 validity report.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _lanczos_tridiag(matvec: Callable[[np.ndarray], np.ndarray], v: np.ndarray, m: int):
    """``m``-step Lanczos tridiagonalization of the operator on unit-norm start ``v``.

    Returns the ``(k, k)`` symmetric tridiagonal ``T`` (``k ≤ m``, shorter on early breakdown).
    """
    n = v.size
    m = min(m, n)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    v_prev = np.zeros(n)
    v_cur = v / (np.linalg.norm(v) + 1e-300)
    w = matvec(v_cur)
    alpha[0] = float(w @ v_cur)
    w = w - alpha[0] * v_cur
    k = m
    for j in range(1, m):
        b = float(np.linalg.norm(w))
        if b < 1e-12:
            k = j
            break
        beta[j] = b
        v_prev, v_cur = v_cur, w / b
        w = matvec(v_cur)
        alpha[j] = float(w @ v_cur)
        w = w - alpha[j] * v_cur - beta[j] * v_prev
    T = np.diag(alpha[:k])
    if k > 1:
        off = beta[1:k]
        T = T + np.diag(off, 1) + np.diag(off, -1)
    return T


def stochastic_logdet(
    matvec: Callable[[np.ndarray], np.ndarray], n: int, *,
    n_probes: int = 16, lanczos_steps: int = 15, seed: int = 0,
    floor: float = 1e-12,
) -> float:
    """Stochastic Lanczos-quadrature estimate of ``log det A`` for a PD operator ``A`` (§V.3).

    ``matvec(x) = A @ x`` is the only access to ``A``. Increasing ``n_probes`` lowers the
    variance (∝ 1/n_probes); ``lanczos_steps`` controls the quadrature accuracy. Eigenvalues of
    the Lanczos tridiagonal are floored at ``floor`` for numerical safety.
    """
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_probes)
    for p in range(n_probes):
        z = rng.choice(np.array([-1.0, 1.0]), size=n)  # Rademacher, ‖z‖² = n
        T = _lanczos_tridiag(matvec, z, lanczos_steps)
        theta, U = np.linalg.eigh(T)
        tau = U[0, :] ** 2                              # (e₁ᵀ eigenvectors)²
        theta = np.clip(theta, floor, None)
        # zᵀ log(A) z ≈ ‖z‖² · Σ τ_k log θ_k  (Lanczos started from z/‖z‖; ‖z‖² = n)
        estimates[p] = n * float(np.sum(tau * np.log(theta)))
    return float(estimates.mean())


def stochastic_logdet_dense(A: np.ndarray, **kwargs) -> float:
    """Convenience wrapper: :func:`stochastic_logdet` for a dense symmetric-PD matrix ``A``."""
    A = np.asarray(A, dtype=np.float64)
    return stochastic_logdet(lambda x: A @ x, A.shape[0], **kwargs)


__all__ = ["stochastic_logdet", "stochastic_logdet_dense"]
