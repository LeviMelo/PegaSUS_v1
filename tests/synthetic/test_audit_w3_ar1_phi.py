"""ADVERSARIAL AUDIT — W3 AR(1) temporal-whitening coefficient (temporal.py, _ar1_phi).

CONTRACT (§III.4): _ar1_phi must estimate TEMPORAL persistence of one variable's (S,T)
slice — the lag-1 dependence WITHIN each spatial unit's time series. On a real panel,
municipalities differ in baseline level (the norm, not the exception). Such between-unit
level variance is a spatial fixed effect, NOT temporal persistence.

BUG: the pre-fix estimator pooled ALL (space,time) lag-1 pairs into flat vectors and
centered by ONE global mean. With strong per-unit level differences, adjacent-in-time
pairs (z_{s,t-1}, z_{s,t}) both sit near unit s's own level, far from the global mean, so
their centered product is large and positive for EVERY unit — the cross-unit level spread
manufactures phi -> ~1 even when each unit's series is iid over time. Over-differencing
then destroys real signal.

FIX: demean per spatial unit before forming lag-1 pairs, i.e.
    phi = corr(z_{s,t} - mean_s, z_{s,t-1} - mean_s) pooled over units.

Test (a) FAILS on the old pooled-global-mean code (phi ~ 0.95) and PASSES on the fix
(phi ~ 0). Test (b) confirms genuine AR(1) persistence is still recovered.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.temporal import _ar1_phi


def _pooled_global_mean_phi(Z_var: np.ndarray) -> float:
    """The OLD (broken) estimator: pool all lag-1 pairs, center by one global mean.
    Reproduced here to demonstrate it fails test (a); asserted, not imported."""
    cur = Z_var[:, 1:]
    prev = Z_var[:, :-1]
    pair = np.isfinite(cur) & np.isfinite(prev)
    a = cur[pair] - cur[pair].mean()
    b = prev[pair] - prev[pair].mean()
    denom = float(np.sqrt((a @ a) * (b @ b)))
    if denom <= 1e-12:
        return 0.0
    return float((a @ b) / denom)


def _white_with_levels(S=60, T=120, spread=8.0, seed=0) -> np.ndarray:
    """Temporally-WHITE panel (iid over time within each unit) with STRONG per-unit levels."""
    rng = np.random.default_rng(seed)
    levels = rng.uniform(-spread, spread, size=(S, 1))  # each unit its own baseline
    noise = rng.standard_normal((S, T))                 # iid over t => zero temporal corr
    return levels + noise


def _ar1_same_level(S=60, T=120, phi=0.7, seed=1) -> np.ndarray:
    """Genuine within-unit AR(1) with phi=0.7, all units share the SAME (zero) level."""
    rng = np.random.default_rng(seed)
    Z = np.zeros((S, T))
    Z[:, 0] = rng.standard_normal(S)
    sd = np.sqrt(1.0 - phi * phi)
    for t in range(1, T):
        Z[:, t] = phi * Z[:, t - 1] + sd * rng.standard_normal(S)
    return Z


def test_old_estimator_fails_on_white_data_with_levels() -> None:
    """Demonstrate the defect: on temporally-white data with per-unit level differences the
    OLD pooled-global-mean estimator reports strong (spurious) persistence."""
    Z = _white_with_levels()
    old = _pooled_global_mean_phi(Z)
    assert old > 0.8, f"expected the broken estimator to fabricate phi~1, got {old:.3f}"


def test_within_unit_phi_zero_on_white_data_with_levels() -> None:
    """THE FIX: temporally-white data + strong per-unit levels must give phi ~ 0.
    Between-unit level variance is a spatial fixed effect, not temporal persistence."""
    Z = _white_with_levels()
    phi = _ar1_phi(Z)
    assert abs(phi) < 0.3, f"within-unit AR(1) must be ~0 on white data, got {phi:.3f}"


def test_within_unit_phi_recovers_true_ar1() -> None:
    """Genuine AR(1) persistence (phi=0.7) must still be recovered within tolerance."""
    Z = _ar1_same_level(phi=0.7)
    phi = _ar1_phi(Z)
    assert abs(phi - 0.7) < 0.1, f"expected phi~0.7, got {phi:.3f}"


def test_within_unit_phi_handles_missing_per_unit() -> None:
    """NaNs are handled per unit: a unit with <2 observed points is skipped; observed
    consecutive pairs still contribute. phi stays finite and in-range."""
    Z = _ar1_same_level(phi=0.7, seed=5).copy()
    Z[0, :] = np.nan                 # a fully-missing unit is dropped
    Z[1, 3:] = np.nan               # a unit with a single observed point is dropped
    Z[2, ::2] = np.nan              # scattered gaps: fewer usable pairs, still valid
    phi = _ar1_phi(Z)
    assert np.isfinite(phi) and -1.0 < phi < 1.0
    assert abs(phi - 0.7) < 0.15
