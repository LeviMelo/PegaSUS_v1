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


def negative_control_break_fraction(
    series_by_var, shock_index: int, *, exclude: set[str], min_level_t: float = 3.0,
) -> tuple[float, int]:
    """§IV negative-control outcomes: the fraction of NEGATIVE-CONTROL series (every variable
    except the edge's endpoints) that ALSO show a significant level change at the SAME ``shock_index``.

    A genuine interruption effect is *specific* to the outcome; if many unrelated series jump at the
    same time, the break is a **common shock** (a confounder affecting everything), not the edge's
    effect — so the ITS attribution should be vetoed. Returns ``(fraction, n_tested)``."""
    controls = [v for v in series_by_var if v not in exclude]
    n_break = n_tested = 0
    for v in controls:
        s = np.asarray(series_by_var[v], dtype=float)
        s = s[np.isfinite(s)]
        if s.size < 10 or not (1 <= shock_index < s.size - 1):
            continue
        n_tested += 1
        if abs(interrupted_time_series(s, shock_index).level_t) >= min_level_t:
            n_break += 1
    return (float(n_break) / n_tested if n_tested else 0.0), n_tested


def escalate_rung2_its(records, series_by_var, *, min_level_t: float = 3.0,
                       nc_veto_fraction: float = 0.5, nc_min_controls: int = 3):
    """Rung-2 quasi-experimental escalation (§IV): for each already-directed (Rung-1) edge
    whose TARGET series has a detected structural break, run an interrupted-time-series at the
    break; if the level change is significant (``|level_t| ≥ min_level_t``) promote the edge to
    Rung 2 and annotate the ITS evidence. This is the machine-checkable auto-trigger ("ITS
    around detected structure"); a validated external shock date remains an expert refinement.
    Edges not yet directed (Rung 0/None) are left untouched — the ladder only escalates upward.
    ``series_by_var[var]`` is that variable's aggregate time series (length T)."""
    from dataclasses import replace as _replace

    import numpy as _np

    out = []
    for r in records:
        if not r.causal_rung or r.causal_rung < 1:
            out.append(r)
            continue
        series = series_by_var.get(r.target_var)
        if series is None:
            out.append(r)
            continue
        series = _np.asarray(series, dtype=float)
        series = series[_np.isfinite(series)]
        if series.size < 10:
            out.append(r)
            continue
        brk = detect_structural_break(series)
        if brk is None:
            out.append(r)
            continue
        its = interrupted_time_series(series, brk)
        if abs(its.level_t) < min_level_t:
            out.append(r)
            continue
        # §IV negative-control veto: if the break also fires on many unrelated series it is a
        # common shock, not this edge's effect — do NOT promote to Rung 2; keep Rung 1 + record.
        nc_frac, nc_n = negative_control_break_fraction(
            series_by_var, brk, exclude={r.source_var, r.target_var}, min_level_t=min_level_t)
        if nc_n >= nc_min_controls and nc_frac >= nc_veto_fraction:
            out.append(_replace(
                r, warnings=r.warnings + (
                    f"rung2_vetoed_negative_control_common_shock:{nc_frac:.2f}",
                    f"rung2_its_shock_t{brk}"),
            ))
            continue
        out.append(_replace(
            r, causal_rung=2,
            causal_assumptions=tuple(dict.fromkeys(
                r.causal_assumptions + ("interrupted_time_series", "negative_control_outcomes"))),
            warnings=r.warnings + (
                f"rung2_its_shock_t{brk}", f"rung2_its_level_t_{its.level_t:.2f}",
                f"rung2_negative_control_clear:{nc_frac:.2f}"),
        ))
    return out


__all__ = [
    "ITSResult", "interrupted_time_series", "detect_structural_break",
    "DiDResult", "difference_in_differences", "escalate_rung2_its",
    "negative_control_break_fraction",
]
