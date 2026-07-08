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


__all__ = ["temporal_whiten"]
