"""LDO Layer 2 — sparse + low-rank latent-variable decomposition (MSD-II §II.6.1, MII-LDO-03).

Decompose the variable-dependency operator ``Ω_var = S − L`` with ``S`` sparse
(the direct + lagged links) and ``L`` low-rank PSD (shared latent drivers — the
ST-DFM factors; e.g. an epidemic wave co-moving many variables). This is the
Chandrasekaran–Parrilo–Willsky latent-variable graphical decomposition, solved by
ADMM:

    min_{S,L}  -logdet(S − L) + tr((S − L) C) + λ₁‖S‖₁,off + λ₂ tr(L),   L ⪰ 0

Variables that co-move only through a common factor load on ``L`` and are reported
as ``latent_shared`` (a confounded pair), NOT as dense direct edges in ``S`` — so
``S`` recovers the direct structure *net of* the shared driver. ``L``'s eigenfactors
are the latent drivers (the ST-DFM factors as Layer 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SparseLowRankFit:
    S: np.ndarray                       # sparse direct precision
    L: np.ndarray                       # low-rank PSD latent component
    precision: np.ndarray               # Ω_var = S - L
    factor_loadings: np.ndarray         # (p x r) loadings of the recovered latent factors
    factor_values: np.ndarray           # (r,) eigenvalues of L
    converged: bool
    iterations: int
    direct_edges: list[tuple[int, int, float]] = field(default_factory=list)   # (i,j,partial_corr) from S
    latent_shared: list[tuple[int, int, float]] = field(default_factory=list)  # (i,j,shared_loading) from L


def _soft_threshold_offdiag(M: np.ndarray, tau: float) -> np.ndarray:
    out = np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)
    np.fill_diagonal(out, np.diag(M))  # do not shrink the diagonal
    return out


def _prox_neg_logdet(M: np.ndarray, rho: float) -> np.ndarray:
    """argmin_R -logdet R + (rho/2)||R - M||^2  (R symmetric PD) via eigenvalues."""
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    d = (vals + np.sqrt(vals**2 + 4.0 / rho)) / 2.0
    return (vecs * d) @ vecs.T


def _psd_project_shifted(M: np.ndarray, shift: float) -> np.ndarray:
    """PSD projection of (M - shift*I): eigen-clip at 0."""
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M - shift * np.eye(M.shape[0]))
    vals = np.clip(vals, 0.0, None)
    return (vecs * vals) @ vecs.T


def fit_sparse_plus_lowrank(
    emp_cov: np.ndarray,
    *,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-5,
    edge_threshold: float = 0.05,
    loading_threshold: float = 0.3,
) -> SparseLowRankFit:
    """LVGLASSO ADMM: ``Ω = S - L`` from an empirical covariance."""
    p = emp_cov.shape[0]
    C = 0.5 * (emp_cov + emp_cov.T) + 1e-4 * np.eye(p)
    S = np.eye(p)
    L = np.zeros((p, p))
    U = np.zeros((p, p))
    R = np.eye(p)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # R-step: prox of -logdet + linear term.
        R = _prox_neg_logdet(S - L - U - C / rho, rho)
        # S-step: soft-threshold off-diagonal.
        S = _soft_threshold_offdiag(R + L + U, lambda1 / rho)
        # L-step: PSD projection with trace shrink.
        L = _psd_project_shifted(S - R - U, lambda2 / rho)
        # Dual update.
        primal = R - (S - L)
        U = U + primal
        if np.linalg.norm(primal) / max(1.0, np.linalg.norm(R)) < tol:
            converged = True
            break

    precision = S - L
    # Direct edges: partial correlations from S.
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    partial = -S / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    direct_edges = [
        (i, j, float(partial[i, j]))
        for i in range(p) for j in range(i + 1, p)
        if abs(partial[i, j]) >= edge_threshold
    ]
    direct_edges.sort(key=lambda e: abs(e[2]), reverse=True)

    # Latent factors: eigen-decomposition of L.
    vals, vecs = np.linalg.eigh(L)
    keep = vals > max(1e-6, 1e-3 * vals.max() if vals.size else 0.0)
    factor_values = vals[keep][::-1]
    factor_loadings = vecs[:, keep][:, ::-1]

    # latent_shared pairs: variables both loading strongly on a common factor.
    latent_shared: list[tuple[int, int, float]] = []
    for f in range(factor_loadings.shape[1]):
        load = factor_loadings[:, f]
        strong = [i for i in range(p) if abs(load[i]) >= loading_threshold]
        for a_i in range(len(strong)):
            for b_i in range(a_i + 1, len(strong)):
                i, j = strong[a_i], strong[b_i]
                latent_shared.append((i, j, float(min(abs(load[i]), abs(load[j])))))
    # keep the strongest shared loading per pair
    best: dict[tuple[int, int], float] = {}
    for i, j, v in latent_shared:
        key = (i, j)
        best[key] = max(best.get(key, 0.0), v)
    latent_shared = sorted(((i, j, v) for (i, j), v in best.items()), key=lambda e: e[2], reverse=True)

    return SparseLowRankFit(
        S=S, L=L, precision=precision,
        factor_loadings=factor_loadings, factor_values=factor_values,
        converged=converged, iterations=it,
        direct_edges=direct_edges, latent_shared=latent_shared,
    )


__all__ = ["SparseLowRankFit", "fit_sparse_plus_lowrank"]
