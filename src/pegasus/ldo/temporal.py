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
    """Pooled lag-1 autocorrelation of one variable's (S,T) slice over pairs where both
    ``z_t`` and ``z_{t-1}`` are observed. 0 when too few pairs or degenerate variance."""
    T = Z_var.shape[1]
    if T < 2:
        return 0.0
    cur = Z_var[:, 1:]
    prev = Z_var[:, :-1]
    pair = np.isfinite(cur) & np.isfinite(prev)
    if int(pair.sum()) < 3:
        return 0.0
    a = cur[pair]
    b = prev[pair]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((a @ b) / denom, -_PHI_CAP, _PHI_CAP))


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
