"""Missing-aware covariance for the LDO (MII-LDO robustness fix).

Real epidemiological panels are sparse: no cell has every variable observed
(complete-case is empty), and imputing missing→0 fabricates a spurious shared
factor that collapses every relationship into ``latent_shared``. The correct
estimator is the **pairwise-complete** covariance — each entry from the cells where
*both* variables are observed — projected to the nearest correlation matrix so the
sparse+low-rank precision estimator receives a valid PSD input.

Variables whose coverage or pairwise overlap is too small are dropped (degenerate
columns), and the surviving index is returned so edges map back to variable names.

§V.4 streaming sufficient statistics (scope boundary): the pairwise moments (``M@M.T``,
``X0@M.T``, ``X0@X0.T``) are the ``XᵀX``/``Xᵀy`` accumulators §V.4 names, computed here over an
in-RAM ``(features × cells)`` matrix. This is exact and fits at the national determinant scale
(``p·(K+1)`` features × ``S·T`` cells is bounded, and the §II.10 envelope guard *refuses* rather
than OOMs when a configuration would not fit). An **out-of-core** accumulation (streaming the same
BLAS products over ``scan_parquet`` row-groups so the cell matrix never fully materializes) is the
beyond-RAM extension of §V.4 — the moment algebra above is already in the streamable form; only the
driver would change. It is not needed for the current national runs and is a documented ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairwiseCovariance:
    correlation: np.ndarray          # (q x q) PSD correlation over kept variables
    kept: tuple[int, ...]            # indices into the original variable axis
    min_overlap: int
    coverage: np.ndarray             # per-kept-variable observed-sample count


def _nearest_correlation(C: np.ndarray, *, floor: float = 1e-3) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD correlation matrix."""
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, floor, None)
    psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    return psd / np.outer(d, d)


def pairwise_correlation(
    samples: np.ndarray,
    *,
    min_coverage: int = 30,
    min_overlap: int = 20,
) -> PairwiseCovariance:
    """Pairwise-complete correlation of a (features × samples) matrix with NaNs.

    ``samples[i]`` is variable ``i`` over all cells (NaN where unobserved).

    Fully vectorized: the per-pair complete-case moments are assembled as a handful
    of BLAS matrix products over the mask ``M`` and the NaN-zeroed data ``X0`` rather
    than a Python double-loop over the ``O(q^2)`` feature pairs (which dominated the
    LDO's per-fit cost, run once per stability subsample). For a pair ``(a,b)`` with
    overlap mask, ``mean_a = (X0 @ M^T)/n``, ``var_a = (X0^2 @ M^T)/n - mean_a^2``,
    ``cov = (X0 @ X0^T)/n - mean_a mean_b`` — the 1/n (or 1/(n-1)) cancels in the
    correlation ratio, so this is numerically identical to per-pair ``np.corrcoef``.
    """
    F, n = samples.shape
    finite = np.isfinite(samples)
    coverage = finite.sum(axis=1)
    # nanstd only over covered rows (avoid all-NaN warnings); std==0 columns are dropped.
    stds = np.zeros(F, dtype=np.float64)
    covered = coverage > 0
    if covered.any():
        stds[covered] = np.nanstd(np.where(finite[covered], samples[covered], np.nan), axis=1)
    kept = [i for i in range(F) if coverage[i] >= min_coverage and stds[i] > 1e-9]
    q = len(kept)
    if q == 0:
        return PairwiseCovariance(correlation=np.zeros((0, 0)), kept=(), min_overlap=min_overlap, coverage=coverage[kept])

    X = samples[kept]                              # (q, n), may contain NaN
    M = np.isfinite(X).astype(np.float64)          # (q, n) overlap mask
    X0 = np.where(M > 0.0, X, 0.0)                  # (q, n) NaN -> 0

    n_ab = M @ M.T                                 # (q, q) pairwise overlap counts
    Sx = X0 @ M.T                                  # [a,b] = sum_t x_a * M_b  (sum of a over overlap)
    Sxx = (X0 * X0) @ M.T                           # [a,b] = sum_t x_a^2 * M_b
    Cxy = X0 @ X0.T                                # [a,b] = sum_t x_a * x_b  (overlap only; x=0 off-mask)

    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(n_ab > 0, 1.0 / n_ab, 0.0)
        mean_a = Sx * inv                          # mean of a over overlap-with-b
        mean_b = Sx.T * inv                        # mean of b over overlap-with-a
        var_a = Sxx * inv - mean_a * mean_a
        var_b = Sxx.T * inv - mean_b * mean_b
        cov = Cxy * inv - mean_a * mean_b
        denom = np.sqrt(np.clip(var_a, 0.0, None) * np.clip(var_b, 0.0, None))
        corr = np.where(denom > 1e-18, cov / denom, 0.0)

    # Enforce the guards the loop applied: enough overlap, finite, degenerate-variance -> 0.
    valid = (n_ab >= min_overlap) & (var_a > 1e-18) & (var_b > 1e-18) & np.isfinite(corr)
    C = np.where(valid, corr, 0.0)
    np.fill_diagonal(C, 1.0)
    C = 0.5 * (C + C.T)

    return PairwiseCovariance(
        correlation=_nearest_correlation(C),
        kept=tuple(kept),
        min_overlap=min_overlap,
        coverage=coverage[kept],
    )


def _sqrt_matvec(Q_sparse, v: np.ndarray, *, m: int = 40) -> np.ndarray:
    """``Q^{1/2} @ v`` for sparse SPD ``Q``, matrix-free via Lanczos (§V.4).

    Builds an ``m``-step Krylov basis with ``m`` sparse matvecs, then evaluates the
    matrix square root on the small ``m×m`` tridiagonal — never factorizes or densifies
    ``Q``. Used only to center the whitened features (needs one vector, ``1``)."""
    from scipy.linalg import eigh_tridiagonal

    n = v.shape[0]
    beta0 = float(np.linalg.norm(v))
    if beta0 == 0.0:
        return v.copy()
    m = min(m, n)
    V = np.zeros((n, m), dtype=np.float64)
    alphas: list[float] = []
    betas: list[float] = []
    V[:, 0] = v / beta0
    w = Q_sparse @ V[:, 0]
    a = float(V[:, 0] @ w)
    alphas.append(a)
    w = w - a * V[:, 0]
    j = 1
    while j < m:
        b = float(np.linalg.norm(w))
        if b < 1e-12:
            break
        betas.append(b)
        V[:, j] = w / b
        w = Q_sparse @ V[:, j]
        a = float(V[:, j] @ w)
        alphas.append(a)
        w = w - a * V[:, j] - b * V[:, j - 1]
        j += 1
    k = len(alphas)
    evals, evecs = eigh_tridiagonal(np.array(alphas), np.array(betas)) if k > 1 else (np.array(alphas), np.array([[1.0]]))
    fT_e1 = evecs @ (np.sqrt(np.clip(evals, 0.0, None)) * evecs[0, :])
    return beta0 * (V[:, :k] @ fT_e1)


def whitened_lagged_correlation(
    Z: np.ndarray,
    Q_sparse,
    K: int,
    *,
    min_coverage: int = 30,
) -> PairwiseCovariance:
    """Spatially-whitened variable×lag correlation via the sparse metric ``Q`` (§V.2/§V.4).

    The GMRF-whitened cross-covariance of two lag features equals ``featᵢᵀ Q featⱼ``
    (for any whitener ``W`` with ``Wᵀ W = Q``, the correlation depends only on ``Q``),
    so it is computed **matrix-free** by sparse matvecs ``Q @ featₜ`` accumulated over
    time slices — never forming a dense ``S×S`` whitener or its ``O(S³)`` square root.
    Result is identical (to numerical precision) to whitening ``Z`` by
    ``Σ_space^{-1/2}=(κI+L_W)^{1/2}`` and then correlating; peak memory is ``O(F·S)``,
    not ``O(S²)``. Missing cells are imputed with the Gaussian margin mean (0) — a dense
    spatial operator cannot honour per-cell missingness (§III.4 whitened estimator).

    ``Z`` is ``(p, S, T)``; feature ``f = lag·p + var``; returns the same
    :class:`PairwiseCovariance` contract as :func:`pairwise_correlation`.
    """
    p, S, T = Z.shape
    T_eff = T - K
    if T_eff <= 0:
        raise ValueError(f"need T>{K} time points for lag order K={K}; got T={T}")
    Zc = np.where(np.isfinite(Z), Z, 0.0)
    finite = np.isfinite(Z)
    F = p * (K + 1)

    # observed-cell count per feature (from the true mask, before impute)
    obs = np.empty(F, dtype=np.int64)
    for lag in range(K + 1):
        obs[lag * p:(lag + 1) * p] = finite[:, :, K - lag: T - lag].sum(axis=(1, 2))

    # Gram matrix G[f,g] = sum_cells whitened(feat_f)·whitened(feat_g) = sum_t feat_tᵀ Q feat_t
    # accumulated slice-by-slice so only an (F,S) block is held, never (F,T·S) or (S,S) dense.
    # R accumulates each feature's spatial sum-over-time, for mean-centering below.
    G = np.zeros((F, F), dtype=np.float64)
    R = np.zeros((F, S), dtype=np.float64)
    Ft = np.empty((F, S), dtype=np.float64)
    for t in range(K, T):
        for lag in range(K + 1):
            Ft[lag * p:(lag + 1) * p] = Zc[:, :, t - lag]
        G += Ft @ (Q_sparse @ Ft.T)  # Q@Ft.T is a sparse (S×S)·(S×F) matvec
        R += Ft

    # Mean-center in the WHITENED space so this equals the Pearson correlation of
    # Σ_space^{-1/2}·Z (not the uncentered second moment). The whitened-feature mean is
    # (1/n)·hᵀ·rowsum with h = Q^{1/2}·1 (Lanczos, matrix-free); G_centered = G − H Hᵀ/n.
    n_cells = S * T_eff
    h = _sqrt_matvec(Q_sparse, np.ones(S, dtype=np.float64))
    H = R @ h
    G = G - np.outer(H, H) / n_cells

    diag = np.clip(np.diag(G), 0.0, None)
    kept = [f for f in range(F) if obs[f] >= min_coverage and diag[f] > 1e-9]
    q = len(kept)
    if q == 0:
        return PairwiseCovariance(correlation=np.zeros((0, 0)), kept=(), min_overlap=0, coverage=np.zeros(0))
    Gk = G[np.ix_(kept, kept)]
    d = np.sqrt(np.clip(np.diag(Gk), 1e-12, None))
    corr = Gk / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    corr = 0.5 * (corr + corr.T)
    return PairwiseCovariance(
        correlation=_nearest_correlation(corr),
        kept=tuple(kept),
        min_overlap=0,
        coverage=np.array([obs[f] for f in kept], dtype=np.int64),
    )


__all__ = ["PairwiseCovariance", "pairwise_correlation", "whitened_lagged_correlation"]
