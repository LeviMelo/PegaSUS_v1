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
    weights: np.ndarray | None = None,
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

    ``weights`` (the §II.6.1 ObservationReliability contract, ``(F, n)`` in [0,1], 0
    where unobserved) makes each cell's contribution proportional to its reliability:
    a pair ``(a,b)``'s cell ``t`` carries effective weight ``w_a[t]·w_b[t]``, so a
    reconstructed/broadcast cell (``W<1``) informs the moment less than a directly
    observed one. The weighted moments are ``n_ab = W_a·W_b``, ``Sx = (X0∘W_a)·W_b``,
    ``Sxx = (X0²∘W_a)·W_b``, ``Cxy = (X0∘W_a)·(X0∘W_b)`` — with ``weights=None`` (⇒
    ``W = M`` the binary mask) every product collapses to the unweighted form above,
    so the estimate is **byte-identical** when no reliability is supplied. Feature
    inclusion (``kept``) and the ``min_overlap`` guard stay on the RAW mask, so
    reliability reshapes the correlation *values* without changing which
    variables/pairs are considered.
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
    # Reliability weight: W = M (binary) when unsupplied → byte-identical to the
    # unweighted moments; else the per-cell reliability, forced to 0 off-mask so a
    # NaN cell can never carry weight (mirrors the assembly-side W[nan]=0 invariant).
    if weights is None:
        Wq = M
    else:
        Wq = np.clip(np.asarray(weights, dtype=np.float64)[kept], 0.0, None) * M

    n_raw = M @ M.T                                # (q, q) RAW overlap counts (govern min_overlap)
    n_ab = Wq @ Wq.T                               # (q, q) reliability-weighted effective pair weights
    XW = X0 * Wq                                   # (q, n) reliability-scaled data
    Sx = XW @ Wq.T                                 # [a,b] = sum_t (w_a x_a) w_b
    Sxx = (X0 * XW) @ Wq.T                          # [a,b] = sum_t (w_a x_a^2) w_b
    Cxy = XW @ XW.T                                # [a,b] = sum_t (w_a x_a)(w_b x_b)

    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.where(n_ab > 0, 1.0 / n_ab, 0.0)
        mean_a = Sx * inv                          # weighted mean of a over overlap-with-b
        mean_b = Sx.T * inv                        # weighted mean of b over overlap-with-a
        var_a = Sxx * inv - mean_a * mean_a
        var_b = Sxx.T * inv - mean_b * mean_b
        cov = Cxy * inv - mean_a * mean_b
        denom = np.sqrt(np.clip(var_a, 0.0, None) * np.clip(var_b, 0.0, None))
        corr = np.where(denom > 1e-18, cov / denom, 0.0)

    # Enforce the guards the loop applied: enough RAW overlap, finite, degenerate-variance -> 0.
    valid = (n_raw >= min_overlap) & (var_a > 1e-18) & (var_b > 1e-18) & np.isfinite(corr)
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
    weights: np.ndarray | None = None,
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

    ``weights`` (the §II.6.1 ObservationReliability contract, ``(p, S, T)`` in [0,1], 0
    where unobserved) reliability-weights each feature-cell in the whitened Gram, so a
    reconstructed/broadcast cell contributes proportionally less to every cross-moment.

    Two regimes. **Uniform / absent** (``weights=None`` or the observed weights vary by
    <1e-9): reliability carries no differential information and correlation is
    scale-invariant, so the estimate is byte-identical to the unweighted whitened path —
    features scaled by the constant weight, mean-centered by the constant ``n_cells``. **Non-
    uniform:** a constant-``n_cells`` centering of ``W``-scaled features subtracts (weighted
    sum)/(raw count), NOT a weighted mean, so the shared reliability-pattern×data-mean term
    survives and two independent variables acquire a spurious edge (audit W1). The fix is a
    proper weighted covariance: center each feature by its WEIGHTED mean
    ``m_v = Σ(W_v·Z_v)/ΣW_v`` over observed cells, then carry the weight as ``√W`` on each
    side (a weighted second moment ``Σ W (x−m)(y−m)`` = ``⟨√W(x−m), √W(y−m)⟩``). The
    raw-weighted-mean centering removes the shared mean up front, so no constant-``n``
    whitened re-centering is applied.

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

    # Non-uniform reliability requires the weighted-covariance path; None or (near-)uniform
    # weights carry no differential info, so keep the current constant-n path byte-identical.
    Wc = None
    nonuniform = False
    if weights is not None:
        Wc = np.where(finite, np.clip(np.asarray(weights, dtype=np.float64), 0.0, None), 0.0)
        obs_w = Wc[finite]
        nonuniform = obs_w.size > 0 and float(obs_w.max() - obs_w.min()) >= 1e-9

    # observed-cell count per feature (from the true mask, before impute) — RAW, governs kept
    obs = np.empty(F, dtype=np.int64)
    for lag in range(K + 1):
        obs[lag * p:(lag + 1) * p] = finite[:, :, K - lag: T - lag].sum(axis=(1, 2))

    if not nonuniform:
        # ── uniform / absent: unchanged constant-n whitened path (byte-identical) ──
        G = np.zeros((F, F), dtype=np.float64)
        R = np.zeros((F, S), dtype=np.float64)
        Ft = np.empty((F, S), dtype=np.float64)
        for t in range(K, T):
            for lag in range(K + 1):
                blk = Zc[:, :, t - lag]
                if Wc is not None:
                    blk = blk * Wc[:, :, t - lag]     # reliability-scale the feature-cell signal
                Ft[lag * p:(lag + 1) * p] = blk
            G += Ft @ (Q_sparse @ Ft.T)  # Q@Ft.T is a sparse (S×S)·(S×F) matvec
            R += Ft
        # Mean-center in the WHITENED space so this equals the Pearson correlation of
        # Σ_space^{-1/2}·Z. Whitened-feature mean = (1/n)·hᵀ·rowsum, h = Q^{1/2}·1 (Lanczos);
        # G_centered = G − H Hᵀ/n.
        n_cells = S * T_eff
        h = _sqrt_matvec(Q_sparse, np.ones(S, dtype=np.float64))
        H = R @ h
        G = G - np.outer(H, H) / n_cells
    else:
        # ── non-uniform: proper weighted covariance ──
        # Weighted mean per feature m_v = Σ(W·Z)/ΣW over the lag-aligned observed cells.
        num = np.zeros(F, dtype=np.float64)   # Σ W·Z
        den = np.zeros(F, dtype=np.float64)   # Σ W
        for lag in range(K + 1):
            sl = slice(K - lag, T - lag)
            num[lag * p:(lag + 1) * p] = (Wc[:, :, sl] * Zc[:, :, sl]).sum(axis=(1, 2))
            den[lag * p:(lag + 1) * p] = Wc[:, :, sl].sum(axis=(1, 2))
        m = np.where(den > 0.0, num / den, 0.0)   # (F,)
        sqrtW = np.sqrt(Wc)
        # Feature at time t: √W ⊙ (Z − m), zeroed off-mask (√W already 0 there).
        G = np.zeros((F, F), dtype=np.float64)
        Ft = np.empty((F, S), dtype=np.float64)
        for t in range(K, T):
            for lag in range(K + 1):
                off = lag * p
                sw = sqrtW[:, :, t - lag]
                Ft[off:off + p] = sw * (Zc[:, :, t - lag] - m[off:off + p, None])
            G += Ft @ (Q_sparse @ Ft.T)

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
