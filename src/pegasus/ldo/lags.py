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
    """(p,S,T) → (p*(K+1), n_samples) preserving NaN; feature f=lag*p+var.

    Fully vectorized via strided slicing: sample column ``c = (t-K)*S + s`` holds
    ``[Z(s,t), Z(s,t-1), …, Z(s,t-K)]`` (block ``lag`` is variables at ``t-lag``).
    The old Python double-loop over ``(t, s)`` did ``S·(T-K)`` ``np.concatenate``
    calls (~140K at national S≈5570, T=25); this builds each lag block as one
    contiguous slice ``Z[:, :, K-lag : T-lag]`` reshaped to ``(p, n)`` — identical
    layout, no per-cell Python.
    """
    p, S, T = Z.shape
    if T <= K:
        raise ValueError(f"need T>{K} time points for lag order K={K}; got T={T}")
    T_eff = T - K
    out = np.empty((p * (K + 1), T_eff * S), dtype=np.float64)
    for lag in range(K + 1):
        # slice t in [K, T) → source t-lag in [K-lag, T-lag); axes (p, T_eff, S) → (p, n)
        block = Z[:, :, K - lag: T - lag]                 # (p, S, T_eff)
        out[lag * p:(lag + 1) * p] = np.swapaxes(block, 1, 2).reshape(p, T_eff * S)
    return out


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
    disease_laplacian: np.ndarray | None = None,
    gamma_temporal: float = 0.0,
    gamma_disease: float = 0.0,
    spatial_whiten: bool = True,
    randomized_factors: bool | None = None,
    float32_bulk: bool = False,
    use_reliability_weights: bool = True,
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
    # §II.6.1 ObservationReliability contract: the per-cell reliability tensor W (provenance +
    # §3.12 field reliability) weights every cell's contribution to the moments — a reconstructed/
    # broadcast cell informs the precision less than a directly observed one. Absent/disabled → the
    # estimators fall back to the unweighted (byte-identical) moments. W is 0 where the value is NaN.
    W = field.W if (use_reliability_weights and getattr(field, "W", None) is not None) else None
    if spatial_whiten and field.Z.shape[1] > 1:
        # Spatial GMRF whitening via the sparse metric (§V.2/§V.4): matrix-free, O(F·S)
        # memory, no dense S×S whitener or O(S³) sqrt — so it scales to national S≈5570.
        from pegasus.ldo.precision import build_spatial_precision_sparse
        Q_space = build_spatial_precision_sparse(field.space_ids, kappa=kappa)
        pw = whitened_lagged_correlation(field.Z, Q_space, K, min_coverage=min_coverage, weights=W)
    else:
        feat = _build_lagged_feature_matrix(field.Z, K)  # (p*(K+1), n)
        w_feat = _build_lagged_feature_matrix(W, K) if W is not None else None
        pw = pairwise_correlation(feat, min_coverage=min_coverage, min_overlap=min_overlap, weights=w_feat)
    kept = pw.kept
    pos = {f: a for a, f in enumerate(kept)}  # feature index → matrix position
    penalty_matrix = None
    if disease_penalty is not None:
        big = tile_penalty_across_lags(disease_penalty, K, lambda1)
        penalty_matrix = big[np.ix_(kept, kept)]
    # §III.4(5) quadratic prior-regularizers: temporal (adjacent-lag) + disease-Laplacian
    # smoothness over the lag-extended feature space, subset to the kept features (matching
    # penalty_matrix). None when both weights are 0 → the CPW fit is byte-identical.
    from pegasus.ldo.lowrank import build_smoothness_operator
    smoothness = build_smoothness_operator(
        p, K, disease_laplacian=disease_laplacian,
        gamma_temporal=gamma_temporal, gamma_disease=gamma_disease,
    )
    if smoothness is not None:
        smoothness = smoothness[np.ix_(kept, kept)]
    # §V.1 float32-bulk policy: store the ADMM iterates in float32 (with float64 reductions
    # + condition-number escalation) when the compute envelope calls for it; float64 otherwise.
    fit = fit_sparse_plus_lowrank(
        pw.correlation, lambda1=lambda1, lambda2=lambda2,
        edge_threshold=edge_threshold, penalty_matrix=penalty_matrix,
        smoothness_operator=smoothness, randomized_factors=randomized_factors,
        work_dtype=np.float32 if float32_bulk else np.float64,
    )

    S = fit.S
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    partial = -S / np.outer(d, d)

    # Scatter the kept-feature partial correlations into the full F×F feature grid (0 for
    # dropped features, exactly what the old ``entry()`` dict-lookup returned). Every
    # readout below is then dense numpy slicing over this grid instead of an O(p²·K)
    # Python loop of dict.get()+float() calls (the readout dominated the per-fit cost at
    # context scale where p is large). ``F = p·(K+1)``.
    F = p * (K + 1)
    partial_full = np.zeros((F, F), dtype=np.float64)
    kept_arr = np.asarray(kept, dtype=np.int64)
    if kept_arr.size:
        partial_full[np.ix_(kept_arr, kept_arr)] = partial

    # Directed lagged links: source i at lag k (>0) → target j at lag 0.
    # curve[j, lag, i] = partial_full[feat(0,j)=j, feat(lag,i)=lag*p+i].
    curve_tensor = partial_full[:p, :].reshape(p, K + 1, p)   # [j, lag, i]
    curve_ji = np.transpose(curve_tensor, (0, 2, 1))          # [j, i, lag]
    mags = np.abs(curve_ji)
    # peak over lags 1..K (lag 0 excluded from the directed peak, as before)
    off = np.eye(p, dtype=bool)                               # i==j mask
    lagged_links: list[LaggedLink] = []
    if K >= 1:
        peak_lag_arr = 1 + np.argmax(mags[:, :, 1:], axis=2)  # [j, i] in 1..K
        peak_mag_arr = np.take_along_axis(mags, peak_lag_arr[:, :, None], axis=2)[:, :, 0]
        peak_mag_arr[off] = 0.0                               # skip i==j (never a self-link)
        for j, i in zip(*np.where(peak_mag_arr >= edge_threshold)):
            j = int(j); i = int(i)
            peak_lag = int(peak_lag_arr[j, i])
            curve = [float(v) for v in curve_ji[j, i]]
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

    # Contemporaneous (undirected) edges from the lag-0 × lag-0 block.
    lag0_block = partial_full[:p, :p]
    iu, ju = np.triu_indices(p, k=1)
    r_vals = lag0_block[iu, ju]
    sel = np.abs(r_vals) >= edge_threshold
    contemporaneous: list[tuple[str, str, float]] = [
        (field.variables[int(a)], field.variables[int(b)], float(r))
        for a, b, r in zip(iu[sel], ju[sel], r_vals[sel])
    ]
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
    # Scatter S over the kept lag-0 base variables; dropped vars keep the identity row/col
    # (diag 1, off-diag 0) so a dropped variable contributes no edge — same as the old
    # per-cell loop, but as two array ops instead of a p² Python double-loop.
    lag0_precision = np.eye(p)
    base_kept = kept_arr[kept_arr < p] if kept_arr.size else np.empty(0, dtype=np.int64)
    if base_kept.size:
        base_pos = np.array([pos[int(f)] for f in base_kept], dtype=np.int64)
        lag0_precision[np.ix_(base_kept, base_kept)] = S[np.ix_(base_pos, base_pos)]

    return LaggedFit(
        variables=field.variables, K=K, fit=fit,
        lagged_links=lagged_links, latent_shared=latent_shared, contemporaneous=contemporaneous,
        lag0_precision=lag0_precision,
    )


__all__ = ["LaggedLink", "LaggedFit", "fit_lagged_links"]
