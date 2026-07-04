"""Rung-2 causal escalation — quasi-experimental leverage (MSD-III §IV, CAUSAL-02).

Triggered by *detected structure* (a dated shock: epidemic onset, policy date, an
interrupted trend), not sprayed everywhere. Two workhorses:

- **Interrupted time series (ITS).** Segmented regression around a shock time ``t0``:
  ``y_t = β0 + β1·t + β2·D_t + β3·(t−t0)·D_t``, ``D_t = 1{t ≥ t0}``. ``β2`` is the
  immediate *level* change at the shock, ``β3`` the change in *slope* — the interruption
  effect, with a t-statistic for each.
- **Difference-in-differences (DiD).** ``(treated_post − treated_pre) − (control_post −
  control_pre)`` — the treatment effect net of a common time trend the control shares.

Each carries its assumptions; nothing is promoted to a causal effect beyond what the
quasi-experimental design licenses. Rung-3 (do-calculus) is expert-invoked only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class ITSResult:
    shock_index: int
    level_change: float      # β2: immediate jump at the shock
    slope_change: float      # β3: change in trend after the shock
    level_t: float           # t-statistic for the level change
    slope_t: float


def interrupted_time_series(y: np.ndarray, shock_index: int) -> ITSResult:
    """Segmented-regression ITS effect of an interruption at ``shock_index``."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if not (1 <= shock_index < n - 1):
        raise ValueError(f"shock_index {shock_index} must be interior to a series of length {n}")
    t = np.arange(n, dtype=np.float64)
    d = (t >= shock_index).astype(np.float64)
    time_since = np.where(d > 0, t - shock_index, 0.0)
    X = np.column_stack([np.ones(n), t, d, time_since])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(1, n - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(sigma2 * np.diag(xtx_inv), _EPS, None))
    return ITSResult(
        shock_index=shock_index,
        level_change=float(beta[2]),
        slope_change=float(beta[3]),
        level_t=float(beta[2] / se[2]),
        slope_t=float(beta[3] / se[3]),
    )


def detect_structural_break(y: np.ndarray, *, min_t: float = 3.0, margin: int = 4) -> int | None:
    """Return the interior index whose ITS level-change is most significant, or ``None``.

    A cheap trigger for ITS: scan candidate breakpoints and keep the one with the largest
    ``|level_t|`` when it clears ``min_t``. ``margin`` keeps a minimum run on each side.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    best_idx, best_t = None, min_t
    for idx in range(margin, n - margin):
        res = interrupted_time_series(y, idx)
        if abs(res.level_t) >= best_t:
            best_idx, best_t = idx, abs(res.level_t)
    return best_idx


@dataclass(frozen=True)
class DiDResult:
    effect: float            # (treated_post − treated_pre) − (control_post − control_pre)
    treated_change: float
    control_change: float


def difference_in_differences(
    treated: np.ndarray, control: np.ndarray, *, pre_mask: np.ndarray, post_mask: np.ndarray
) -> DiDResult:
    """Difference-in-differences effect, netting out the control's common time trend."""
    treated = np.asarray(treated, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    tc = float(treated[post_mask].mean() - treated[pre_mask].mean())
    cc = float(control[post_mask].mean() - control[pre_mask].mean())
    return DiDResult(effect=tc - cc, treated_change=tc, control_change=cc)


__all__ = [
    "ITSResult", "interrupted_time_series", "detect_structural_break",
    "DiDResult", "difference_in_differences",
]
