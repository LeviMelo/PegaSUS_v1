"""WP8 / §V.3 — matrix-free stochastic log-determinant (Hutchinson + Lanczos quadrature).

The Gaussian likelihood's ``log det Ω`` at national scale cannot use an O(n³) eigh; §V.3
prescribes an unbiased probe-count-controlled estimator touching the operator only via matvecs.
This pins that it approximates the exact log-det and that its error shrinks with probe count.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.stochastic import stochastic_logdet, stochastic_logdet_dense


def _pd(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n))
    return B @ B.T / n + np.eye(n)


def test_stochastic_logdet_matches_exact_within_tolerance():
    A = _pd(200)
    exact = float(np.linalg.slogdet(A)[1])
    est = stochastic_logdet_dense(A, n_probes=64, lanczos_steps=25, seed=1)
    assert abs(est - exact) / abs(exact) < 0.03


def test_error_shrinks_with_more_probes():
    A = _pd(200)
    exact = float(np.linalg.slogdet(A)[1])
    err = lambda m: np.mean([abs(stochastic_logdet_dense(A, n_probes=m, lanczos_steps=25, seed=s) - exact)
                             for s in range(6)])
    assert err(64) < err(8)  # variance ∝ 1/probes


def test_matrix_free_via_matvec():
    A = _pd(120, seed=2)
    exact = float(np.linalg.slogdet(A)[1])
    # only the matvec is exposed — never the dense matrix
    est = stochastic_logdet(lambda x: A @ x, A.shape[0], n_probes=48, lanczos_steps=25, seed=3)
    assert abs(est - exact) / abs(exact) < 0.05
