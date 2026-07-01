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

from pegasus.pirs.ldo.margins import GaussianField
from pegasus.pirs.ldo.lowrank import SparseLowRankFit, fit_sparse_plus_lowrank
from pegasus.pirs.ldo.precision import _matrix_sqrt_psd, build_spatial_precision


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


def _spatially_whiten(field: GaussianField, kappa: float) -> np.ndarray:
    p, S, T = field.shape
    Zc = np.where(np.isfinite(field.Z), field.Z, 0.0)
    if S <= 1:
        return Zc
    Q = build_spatial_precision(field.space_ids, kappa=kappa)
    Q_half = _matrix_sqrt_psd(Q)
    out = np.empty_like(Zc)
    for j in range(p):
        out[j] = Q_half @ Zc[j]
    return out


def _build_lagged_samples(Zw: np.ndarray, K: int) -> np.ndarray:
    """(p,S,T) whitened → (n_samples, p*(K+1)); columns [lag0 vars.., lag1.., ..]."""
    p, S, T = Zw.shape
    if T <= K:
        raise ValueError(f"need T>{K} time points for lag order K={K}; got T={T}")
    rows: list[np.ndarray] = []
    for t in range(K, T):
        for s in range(S):
            feat = np.concatenate([Zw[:, s, t - lag] for lag in range(K + 1)])
            rows.append(feat)
    return np.asarray(rows, dtype=np.float64)


def fit_lagged_links(
    field: GaussianField,
    *,
    K: int = 8,
    kappa: float = 1.0,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    edge_threshold: float = 0.05,
) -> LaggedFit:
    """Fit the time-extended precision and read off directed lagged links."""
    p = len(field.variables)
    Zw = _spatially_whiten(field, kappa)
    D = _build_lagged_samples(Zw, K)
    D = (D - D.mean(axis=0)) / np.where(D.std(axis=0) == 0, 1.0, D.std(axis=0))
    emp = np.cov(D, rowvar=False)
    fit = fit_sparse_plus_lowrank(emp, lambda1=lambda1, lambda2=lambda2, edge_threshold=edge_threshold)

    S = fit.S
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    partial = -S / np.outer(d, d)

    # Directed lagged links: source i at lag k (>0) → target j at lag 0.
    lagged_links: list[LaggedLink] = []
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            curve = [float(partial[j, lag * p + i]) for lag in range(K + 1)]
            # peak over lags >= 1 (directed); lag 0 is contemporaneous/undirected
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

    # latent_shared over the lag-0 block (variables co-loading on a common factor).
    latent_shared: list[tuple[str, str, float]] = []
    for i, j, v in fit.latent_shared:
        if i < p and j < p:  # lag-0 variables only
            latent_shared.append((field.variables[i], field.variables[j], v))

    return LaggedFit(variables=field.variables, K=K, fit=fit, lagged_links=lagged_links, latent_shared=latent_shared)


__all__ = ["LaggedLink", "LaggedFit", "fit_lagged_links"]
