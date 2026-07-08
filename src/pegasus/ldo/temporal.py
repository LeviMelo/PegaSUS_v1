"""Temporal pre-whitening for the LDO (§III.4 — the time-axis dual of spatial whitening).

Space is whitened before the precision estimate; time is not. A shared AR(1) trend then
reads as a lagged or contemporaneous edge between two independent variables. Estimate an
AR(1) ``phi`` per variable from lag-1 autocorrelation over the observed cells and apply
``Σ_time^{-1/2}``: ``z_t → (z_t − phi·z_{t-1})/sqrt(1−phi²)`` for ``t≥1`` (``t=0`` kept).
Missing cells are never fabricated — a whitened value needs both ``z_t`` and ``z_{t-1}``
observed, otherwise it stays NaN.
"""

from __future__ import annotations

import numpy as np

_PHI_CAP = 0.98


def _ar1_phi(Z_var: np.ndarray) -> float:
    """WITHIN-UNIT pooled lag-1 AR(1) coefficient of one variable's (S,T) slice.

    Each spatial unit is demeaned by its OWN temporal mean before forming lag-1 pairs, so
    between-unit level differences cannot masquerade as temporal persistence. phi is the
    pooled within-unit lag-1 autocovariance over its within-unit variance. A unit needs
    >=2 observed points to define a mean and contributes only its observed consecutive
    pairs. 0 when too few pairs or degenerate variance."""
    S, T = Z_var.shape
    if T < 2:
        return 0.0
    num = 0.0
    den = 0.0
    npair = 0
    for s in range(S):
        row = Z_var[s]
        obs = np.isfinite(row)
        if int(obs.sum()) < 2:
            continue
        mu = row[obs].mean()
        d = np.where(obs, row - mu, np.nan)
        cur = d[1:]
        prev = d[:-1]
        pair = np.isfinite(cur) & np.isfinite(prev)
        k = int(pair.sum())
        if k == 0:
            continue
        num += float(cur[pair] @ prev[pair])
        den += float(prev[pair] @ prev[pair])
        npair += k
    if npair < 3 or den <= 1e-12:
        return 0.0
    return float(np.clip(num / den, -_PHI_CAP, _PHI_CAP))


def temporal_whiten(
    Z: np.ndarray, *, pooled: bool = False, phi: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) pre-whiten the time axis of a (p,S,T) latent field.

    Returns ``(Z_white, phi_by_var)``. ``phi`` may be supplied to reuse an estimate;
    ``pooled=True`` uses one shared phi (mean of per-variable estimates) across variables.
    """
    Z = np.asarray(Z, dtype=np.float64)
    p, S, T = Z.shape
    if phi is None:
        phi = np.array([_ar1_phi(Z[j]) for j in range(p)], dtype=np.float64)
        if pooled and p > 0:
            phi = np.full(p, float(phi.mean()))
    else:
        phi = np.asarray(phi, dtype=np.float64)
    out = np.full_like(Z, np.nan)
    out[:, :, 0] = Z[:, :, 0]
    if T >= 2:
        cur = Z[:, :, 1:]
        prev = Z[:, :, :-1]
        scale = np.sqrt(np.clip(1.0 - phi * phi, 1e-6, None))[:, None, None]
        w = (cur - phi[:, None, None] * prev) / scale
        w[~(np.isfinite(cur) & np.isfinite(prev))] = np.nan
        out[:, :, 1:] = w
    return out, phi


def detrend_latent_field(Z: np.ndarray, *, degree: int | None = None) -> np.ndarray:
    """Remove a per-(variable, municipality) smooth polynomial time trend from the latent field ``Z``
    (LDO-LAG-CONF-01 / LDO-LAG-STAT-02). Each muni's series is residualized against a low-order poly in
    time, so the latent scale carries the within-muni ANOMALY net of (a) the time-invariant municipal
    baseline — the degree-0 term is the municipality fixed effect, extending the count-margin's per-muni
    baseline to every variable — and (b) the secular/reporting-completeness trend that otherwise makes
    two independently-trending series show a spurious contemporaneous or lagged edge (Yule's nonsense
    correlation). A LOW-order poly is deliberate: it captures the smooth secular trend but CANNOT fit an
    epidemic spike, so anomalies (the actual epidemiological signal — e.g. the 2015 Zika wave) survive.

    ``degree`` auto-scales with the span (2 if ``T≥10``, 1 if ``T≥6``, else 0 = per-muni de-mean only)
    so short panels are not over-fit. Missing cells stay NaN (they are 0-imputed only for the shared
    projection — a smooth low-order fit is barely moved by a few imputed cells, verified immaterial for
    the whitener). Correlations are scale-free, so the variance the detrend removes does not bias edges.
    Vectorized as a residual off the polynomial column space: ``resid = Z − Z·P``, ``P=V(VᵀV)⁻¹Vᵀ``."""
    Z = np.asarray(Z, dtype=np.float64)
    p, S, T = Z.shape
    if T < 3:
        return Z
    if degree is None:
        degree = 2 if T >= 10 else (1 if T >= 6 else 0)
    t = np.arange(T, dtype=np.float64)
    V = np.vander(t, degree + 1)                      # (T, degree+1) polynomial basis
    P = V @ np.linalg.pinv(V)                          # (T, T) projector onto the poly column space
    finite = np.isfinite(Z)
    Zc = np.where(finite, Z, 0.0)
    resid = Zc - np.einsum("pst,tu->psu", Zc, P)       # (p,S,T) residual off the trend, all series at once
    return np.where(finite, resid, np.nan)


def remove_common_trend(Z: np.ndarray, *, k: int = 1) -> np.ndarray:
    """Remove the leading ``k`` SHARED temporal factors from the latent field (LDO-LAG-CONF-01 /
    LDO-LAG-STAT-02) — the surgical alternative to per-series detrending.

    The confounding that manufactures spurious contemporaneous/lagged edges is the trend COMMON across
    variables (the secular epidemiological transition, the SUS reporting-completeness ramp). This
    estimates that common component as the leading right-singular vector(s) of the ``(p×T)`` national-
    mean-per-variable matrix (centered over time) and projects it out of every ``Z_{j,s,·}``. Because it
    removes only the SHARED temporal pattern — not each variable's own low-frequency band — a
    variable-specific lagged relationship (which is orthogonal to the common factor) is preserved, unlike
    the blunter :func:`detrend_latent_field` (verified: planted lag-6 corr 0.73→0.72 here vs a per-series
    detrend that tipped it out of recovery; a shared-trend spurious lag 0.91→0.01). Missing cells stay NaN."""
    Z = np.asarray(Z, dtype=np.float64)
    p, S, T = Z.shape
    if T < 3 or k < 1:
        return Z
    finite = np.isfinite(Z)
    with np.errstate(invalid="ignore"):
        M = np.where(np.isfinite(np.nanmean(np.where(finite, Z, np.nan), axis=1)),
                     np.nanmean(np.where(finite, Z, np.nan), axis=1), 0.0)   # (p,T) national mean/var
    Mc = M - M.mean(axis=1, keepdims=True)
    if not np.any(np.abs(Mc) > 1e-12):
        return Z                                              # no shared temporal structure → no-op
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    G = Vt[: min(k, Vt.shape[0])]                             # (k,T) orthonormal shared temporal factors
    Zc = np.where(finite, Z, 0.0)
    proj = np.einsum("pst,kt->psk", Zc, G)                    # coordinates on the shared factors
    resid = Zc - np.einsum("psk,kt->pst", proj, G)
    return np.where(finite, resid, np.nan)


__all__ = ["temporal_whiten", "detrend_latent_field", "remove_common_trend"]
