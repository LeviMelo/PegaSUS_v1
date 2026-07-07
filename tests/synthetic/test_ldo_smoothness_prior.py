"""WP1 / §III.4(5) — the quadratic prior-regularizers (temporal + disease Laplacian).

Proof of capability for the two prescribed smoothness quadratics `+(1/2)·tr(Sᵀ G S)` that
were missing from the LVGLASSO: temporal (adjacent-lag) and disease-Laplacian (structurally
close diseases get similar precision rows). The spatial GMRF remains the whitening metric;
the BYM varying-coefficient field is WP2.

What is pinned:
- γ=0 (or no operator) is byte-identical to the pure CPW fit — the S/L split is unperturbed.
- The disease-Laplacian quadratic *monotonically* shrinks the precision rows of graph-linked
  variables toward each other as γ grows (the borrow-strength effect), while converging.
- `build_smoothness_operator` lays the operator out in the `f = lag·p + var` feature space.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.lowrank import (
    build_smoothness_operator,
    fit_sparse_plus_lowrank,
    lag_chain_laplacian,
)


def _structured_cov(p: int = 8, n: int = 400, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    common = rng.standard_normal(n)
    Z[:, 0] += 1.2 * common  # variables 0,1 co-move (a disease-neighbour pair)
    Z[:, 1] += 1.1 * common
    Z[:, 2] += 0.9 * Z[:, 5]  # an unrelated direct edge
    return np.cov(Z, rowvar=False) + 1e-3 * np.eye(p)


def test_smoothness_operator_none_is_byte_identical() -> None:
    C = _structured_cov()
    base = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.1)
    off = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.1, smoothness_operator=None)
    assert np.array_equal(base.S, off.S)
    assert np.array_equal(base.L, off.L)
    # a γ=0 build returns None → same path
    assert build_smoothness_operator(8, K=2, gamma_temporal=0.0, gamma_disease=0.0) is None


def test_disease_laplacian_quadratic_smooths_linked_rows_monotonically() -> None:
    p = 8
    C = _structured_cov(p=p)
    # disease graph links the co-moving pair (0,1)
    W = np.zeros((p, p))
    W[0, 1] = W[1, 0] = 1.0
    L_D = np.diag(W.sum(axis=1)) - W

    def linked_row_distance(gamma: float) -> tuple[float, bool]:
        G = build_smoothness_operator(p, K=0, disease_laplacian=L_D, gamma_disease=gamma)
        fit = fit_sparse_plus_lowrank(C, lambda1=0.05, lambda2=0.1, smoothness_operator=G)
        mask = np.ones(p, dtype=bool)
        mask[[0, 1]] = False  # compare their connections to the OTHER variables
        return float(np.linalg.norm(fit.S[0][mask] - fit.S[1][mask])), fit.converged

    d0, c0 = linked_row_distance(0.0)
    d_mid, c_mid = linked_row_distance(0.3)
    d_hi, c_hi = linked_row_distance(1.0)
    assert c0 and c_mid and c_hi  # the prox-gradient step converges at every strength
    # borrow-strength: the more we smooth, the more the linked pair's rows agree
    assert d_mid <= d0 + 1e-9
    assert d_hi <= d_mid + 1e-9
    assert d_hi < d0  # a real (not vacuous) effect


def test_temporal_and_operator_layout() -> None:
    # chain Laplacian: rows sum to 0, tridiagonal path graph over K+1 lags
    L = lag_chain_laplacian(3)
    assert L.shape == (4, 4)
    assert np.allclose(L.sum(axis=1), 0.0)
    assert L[0, 0] == 1.0 and L[1, 1] == 2.0  # endpoints degree 1, interior degree 2

    # feature-space operator is (p·(K+1))² and block-structured (f = lag·p + var)
    p, K = 5, 2
    W = np.zeros((p, p)); W[0, 1] = W[1, 0] = 1.0
    L_D = np.diag(W.sum(axis=1)) - W
    G = build_smoothness_operator(p, K, disease_laplacian=L_D, gamma_temporal=0.5, gamma_disease=0.5)
    assert G is not None and G.shape == (p * (K + 1), p * (K + 1))
    assert np.allclose(G, G.T)  # symmetric PSD operator
    assert np.linalg.eigvalsh(G).min() > -1e-9
