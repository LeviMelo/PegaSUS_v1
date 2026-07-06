"""LDO lag extension — directed distributed-lag links (MSD-II §II.6.1, MII-LDO-04).

Extend the variable axis to lags ``0..K``: the time-extended feature at cell
``(s,t)`` is ``[Z(s,t), Z(s,t-1), …, Z(s,t-K)]`` (``p·(K+1)`` dims). Estimating the
sparse+low-rank precision over this extended set exposes the cross-lag blocks: a
nonzero entry linking source ``i`` at lag ``k>0`` to target ``j`` at lag ``0`` is a
**directed lag-``k`` link** ``X_i(t-k) → X_j(t)`` (time licenses the direction). The
profile ``{S^{(k,0)}_{ij}}_{k=0}^K`` is the discovered distributed-lag response
curve — the engine finds the lag; it is not told it.

The shared low-rank component ``L`` still absorbs common drivers (an epidemic
wave), so a co-epidemic is attributed to ``latent_shared`` rather than a spurious
lagged edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pegasus.ldo.covariance import pairwise_correlation, whitened_lagged_correlation
from pegasus.ldo.disease_prior import tile_penalty_across_lags
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.lowrank import SparseLowRankFit, fit_sparse_plus_lowrank


@dataclass
class LaggedLink:
    source: str
    target: str
    peak_lag: int
    peak_partial_correlation: float
    response_curve: list[float]   # {partial corr at lag 0..K}


@dataclass
class LaggedFit:
    variables: tuple[str, ...]
    K: int
    fit: SparseLowRankFit
    lagged_links: list[LaggedLink] = field(default_factory=list)
    latent_shared: list[tuple[str, str, float]] = field(default_factory=list)
    contemporaneous: list[tuple[str, str, float]] = field(default_factory=list)  # (i, j, partial_corr) lag-0
    lag0_precision: np.ndarray | None = None  # (p×p) lag-0 precision aligned to `variables`


def _build_lagged_feature_matrix(Z: np.ndarray, K: int) -> np.ndarray:
    """(p,S,T) → (p*(K+1), n_samples) preserving NaN; feature f=lag*p+var."""
    p, S, T = Z.shape
    if T <= K:
        raise ValueError(f"need T>{K} time points for lag order K={K}; got T={T}")
    cols: list[np.ndarray] = []
    for t in range(K, T):
        for s in range(S):
            cols.append(np.concatenate([Z[:, s, t - lag] for lag in range(K + 1)]))
    return np.asarray(cols, dtype=np.float64).T  # (features, samples)


def fit_lagged_links(
    field: GaussianField,
    *,
    K: int = 8,
    kappa: float = 1.0,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    edge_threshold: float = 0.05,
    min_coverage: int = 30,
    min_overlap: int = 20,
    disease_penalty: np.ndarray | None = None,
    spatial_whiten: bool = True,
) -> LaggedFit:
    """Fit the time-extended precision (missing-aware) and read off directed lagged links.

    The lag-extended features (variable × lag) are correlated pairwise-complete —
    each entry from the cells where both lagged features are observed — so links
    survive the sparsity of real panels instead of collapsing under impute-0.

    ``disease_penalty`` (``p×p``, over the base variables) supplies the disease-axis
    prior (§II.6/§5.2): it is tiled across the lag blocks and subset to the kept
    features so related-disease links (at any lag) get a lower ℓ1 penalty.

    ``spatial_whiten`` (§III.4(3/5), §V.2) removes spatial autocorrelation before the
    precision estimate by GMRF-whitening each variable/time slice across space by
    ``Σ_space^{-1/2} = (κI + L_W)^{1/2}`` (the SpatialWeightGraph Laplacian). Without
    it, ``Ω_var`` is estimated from correlations that still carry spatial
    autocorrelation — the dominant municipal-scale confounder, so two variables that
    merely co-cluster in space read as a spurious direct link. Whitening imputes the
    Gaussian margin mean (0) at missing cells (a dense op cannot honour per-cell
    missingness); where the field's municipalities are absent from the adjacency the
    Laplacian term is empty and whitening reduces to a κ-scaling (a no-op at κ=1).
    """
    p = len(field.variables)
    if spatial_whiten and field.Z.shape[1] > 1:
        # Spatial GMRF whitening via the sparse metric (§V.2/§V.4): matrix-free, O(F·S)
        # memory, no dense S×S whitener or O(S³) sqrt — so it scales to national S≈5570.
        from pegasus.ldo.precision import build_spatial_precision_sparse
        Q_space = build_spatial_precision_sparse(field.space_ids, kappa=kappa)
        pw = whitened_lagged_correlation(field.Z, Q_space, K, min_coverage=min_coverage)
    else:
        feat = _build_lagged_feature_matrix(field.Z, K)  # (p*(K+1), n)
        pw = pairwise_correlation(feat, min_coverage=min_coverage, min_overlap=min_overlap)
    kept = pw.kept
    pos = {f: a for a, f in enumerate(kept)}  # feature index → matrix position
    penalty_matrix = None
    if disease_penalty is not None:
        big = tile_penalty_across_lags(disease_penalty, K, lambda1)
        penalty_matrix = big[np.ix_(kept, kept)]
    fit = fit_sparse_plus_lowrank(
        pw.correlation, lambda1=lambda1, lambda2=lambda2,
        edge_threshold=edge_threshold, penalty_matrix=penalty_matrix,
    )

    S = fit.S
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    partial = -S / np.outer(d, d)

    def entry(feat_a: int, feat_b: int) -> float:
        a, b = pos.get(feat_a), pos.get(feat_b)
        return float(partial[a, b]) if a is not None and b is not None else 0.0

    # Directed lagged links: source i at lag k (>0) → target j at lag 0.
    lagged_links: list[LaggedLink] = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            curve = [entry(j, lag * p + i) for lag in range(K + 1)]  # feat(0,j) vs feat(k,i)
            lag_mags = [(k, abs(curve[k])) for k in range(1, K + 1)]
            if not lag_mags:
                continue
            peak_lag, peak_mag = max(lag_mags, key=lambda kv: kv[1])
            if peak_mag >= edge_threshold:
                lagged_links.append(
                    LaggedLink(
                        source=field.variables[i],
                        target=field.variables[j],
                        peak_lag=peak_lag,
                        peak_partial_correlation=curve[peak_lag],
                        response_curve=curve,
                    )
                )
    lagged_links.sort(key=lambda e: abs(e.peak_partial_correlation), reverse=True)

    # Contemporaneous (undirected) edges from the lag-0 × lag-0 block of S.
    contemporaneous: list[tuple[str, str, float]] = []
    for i in range(p):
        for j in range(i + 1, p):
            r = entry(i, j)  # feat(0,i) vs feat(0,j)
            if abs(r) >= edge_threshold:
                contemporaneous.append((field.variables[i], field.variables[j], r))
    contemporaneous.sort(key=lambda e: abs(e[2]), reverse=True)

    # latent_shared over the lag-0 block: kept features that are lag-0 variables.
    lag0_feature_to_var = {f: f for f in kept if f < p}
    latent_shared: list[tuple[str, str, float]] = []
    for a, b, v in fit.latent_shared:
        fa, fb = kept[a], kept[b]
        if fa in lag0_feature_to_var and fb in lag0_feature_to_var:
            latent_shared.append((field.variables[fa], field.variables[fb], v))

    # Lag-0 precision aligned to the p base variables (feature index of base var i at
    # lag 0 is i). Built through the kept `pos` map so a dropped low-coverage variable
    # defaults to identity (no edge) rather than misindexing a lag>0 feature row — the
    # residual scan requires a p×p block aligned to `variables`; slicing S[:p,:p] by raw
    # position misattributes edges when any lag-0 var is dropped and is undersized (q<p)
    # when many are, silently disabling the scan.
    lag0_precision = np.eye(p)
    for i in range(p):
        ai = pos.get(i)
        if ai is None:
            continue
        for j in range(p):
            aj = pos.get(j)
            if aj is not None:
                lag0_precision[i, j] = S[ai, aj]

    return LaggedFit(
        variables=field.variables, K=K, fit=fit,
        lagged_links=lagged_links, latent_shared=latent_shared, contemporaneous=contemporaneous,
        lag0_precision=lag0_precision,
    )


__all__ = ["LaggedLink", "LaggedFit", "fit_lagged_links"]
