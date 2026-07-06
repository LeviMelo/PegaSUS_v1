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
    """PSD projection of (M - shift*I): eigen-clip at 0.

    ``M - shift·I`` shares its eigenvectors with ``M`` and only shifts the eigenvalues
    by ``-shift``, so the shift is applied to the eigenvalues after a single ``eigh(M)``
    — identical result, without allocating/subtracting a ``shift·I`` matrix every ADMM
    iteration (this runs twice per iteration for up to ``max_iter`` iterations).
    """
    M = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals - shift, 0.0, None)
    return (vecs * vals) @ vecs.T


def _low_rank_factors(
    L: np.ndarray, *, rank_cap: int, randomized: bool, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenfactors of the PSD low-rank component ``L`` (values desc, loadings aligned).

    MSD-III §V.3: the shared-driver factors are few, so at scale the top-``rank_cap``
    eigenpairs are recovered by **randomized SVD** (Halko–Martinsson–Tropp) in O(p²·r)
    instead of a dense O(p³) eigh — bounded error since ``L`` is genuinely low-rank
    (the tail past its numerical rank is zero). This is a post-convergence readout; it
    does not touch the ADMM iteration (or the CPW S/L split), so the exact and
    randomized paths agree to tolerance. Small ``p`` uses the exact eigh.
    """
    p = L.shape[0]
    if randomized and p > 2 * rank_cap and rank_cap >= 1:
        from sklearn.utils.extmath import randomized_svd

        # L is symmetric PSD → its SVD is its eigendecomposition (U singular vectors are
        # eigenvectors, singular values are the non-negative eigenvalues).
        U, s, _ = randomized_svd(L, n_components=min(rank_cap, p - 1), random_state=seed,
                                 n_oversamples=10, n_iter=4)
        return s, U
    vals, vecs = np.linalg.eigh(L)
    return vals[::-1], vecs[:, ::-1]


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
    min_factor_support: int = 3,
    penalty_matrix: np.ndarray | None = None,
    factor_rank_cap: int = 24,
    randomized_factors: bool | None = None,
    seed: int = 0,
) -> SparseLowRankFit:
    """LVGLASSO ADMM: ``Ω = S - L`` from an empirical covariance.

    ``penalty_matrix`` (p×p, symmetric ≥0) overrides the scalar ``lambda1`` with a
    per-pair ℓ1 penalty — the disease-structure prior (§II.6/§5.2): structurally
    related disease-concept variables get a *lower* penalty so their (sparse) edges
    survive, i.e. dependency profiles vary smoothly across the disease hierarchy.
    Absent it, the estimator is the plain scalar-penalty LVGLASSO (a no-op prior).

    ``min_factor_support`` enforces the Chandrasekaran–Parrilo–Willsky **incoherence
    identifiability condition** at readout: a genuine low-rank latent driver must be
    *spread* across variables, so a recovered factor whose strong loadings concentrate
    on fewer than ``min_factor_support`` variables is not a shared driver but a direct
    edge (a rank-1 component on 2 variables is exactly a strong pairwise link). Such
    concentrated factors are dropped from ``latent_shared`` (they surface in ``S`` as
    ``direct_edges``). Without this, one direct edge is double-reported as both a
    ``contemporaneous`` and a ``latent_shared`` link, and there is no fixed ``lambda2``
    that separates the two roles — the collapse the LDO exhibited at every operating
    point.
    """
    p = emp_cov.shape[0]
    C = 0.5 * (emp_cov + emp_cov.T) + 1e-4 * np.eye(p)
    if penalty_matrix is not None:
        penalty_matrix = np.asarray(penalty_matrix, dtype=np.float64)
        if penalty_matrix.shape != (p, p):
            raise ValueError(f"penalty_matrix must be {(p, p)}, got {penalty_matrix.shape}")
        penalty_matrix = np.clip(0.5 * (penalty_matrix + penalty_matrix.T), 0.0, None)
    tau1 = (penalty_matrix if penalty_matrix is not None else lambda1) / rho
    C_over_rho = C / rho            # loop-invariant; was recomputed every ADMM iteration
    l2_over_rho = lambda2 / rho     # loop-invariant shift for the L-step PSD projection
    S = np.eye(p)
    L = np.zeros((p, p))
    U = np.zeros((p, p))
    R = np.eye(p)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # R-step: prox of -logdet + linear term.
        R = _prox_neg_logdet(S - L - U - C_over_rho, rho)
        # S-step: soft-threshold off-diagonal (scalar or disease-informed per-pair penalty).
        S = _soft_threshold_offdiag(R + L + U, tau1)
        # L-step: PSD projection with trace shrink.
        L = _psd_project_shifted(S - R - U, l2_over_rho)
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

    # Latent factors: eigendecomposition of L (randomized SVD at scale, §V.3).
    use_randomized = randomized_factors if randomized_factors is not None else (p > 2 * factor_rank_cap)
    vals, vecs = _low_rank_factors(L, rank_cap=factor_rank_cap, randomized=use_randomized, seed=seed)
    # Keep only factors clearly above the noise floor (2% of the top eigenvalue). Shared
    # drivers are few and large; the fat tail of small PSD-projection eigenvalues emitted
    # spurious latent_shared pairs and made the exact/randomized readouts disagree —
    # dropping it cleans the readout and lets §V.6 exact-certifies-approximate hold.
    keep = vals > max(1e-6, 0.02 * vals.max() if vals.size else 0.0)
    factor_values = vals[keep]
    factor_loadings = vecs[:, keep]

    # latent_shared pairs: variables both loading strongly on a common factor.
    # CPW incoherence gate: a factor supported on < min_factor_support variables is a
    # concentrated (coherent) component — a direct edge, not a shared driver — so it is
    # not reported as latent_shared (it is already recovered in S / direct_edges).
    latent_shared: list[tuple[int, int, float]] = []
    for f in range(factor_loadings.shape[1]):
        load = factor_loadings[:, f]
        strong = [i for i in range(p) if abs(load[i]) >= loading_threshold]
        if len(strong) < min_factor_support:
            continue
        for a_i in range(len(strong)):
            for b_i in range(a_i + 1, len(strong)):
                i, j = strong[a_i], strong[b_i]
                latent_shared.append((i, j, float(min(abs(load[i]), abs(load[j])))))
    # keep the strongest shared loading per pair
    best: dict[tuple[int, int], float] = {}
    for i, j, v in latent_shared:
        key = (i, j)
        best[key] = max(best.get(key, 0.0), v)
    # Mutual exclusion (§III.4.2): a pair reported as a direct edge in S is not also a
    # confounded (latent_shared) pair — one pair, one role. S (net of the shared driver)
    # is authoritative for direct links, so drop any direct-edge pair from latent_shared.
    direct_pairs = {(i, j) for i, j, _ in direct_edges}
    latent_shared = sorted(
        ((i, j, v) for (i, j), v in best.items() if (i, j) not in direct_pairs),
        key=lambda e: e[2], reverse=True,
    )

    return SparseLowRankFit(
        S=S, L=L, precision=precision,
        factor_loadings=factor_loadings, factor_values=factor_values,
        converged=converged, iterations=it,
        direct_edges=direct_edges, latent_shared=latent_shared,
    )


__all__ = ["SparseLowRankFit", "fit_sparse_plus_lowrank"]
