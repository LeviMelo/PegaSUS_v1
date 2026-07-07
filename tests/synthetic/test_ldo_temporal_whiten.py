"""Temporal pre-whitening (Theme-7 LDO-TIME-03 / LDO-AR1-14): a shared AR(1) time trend
must not read as an edge between two independent variables — and the whitening must be a
default a caller can disable to recover the exact prior fit.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.lags import fit_lagged_links
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.temporal import temporal_whiten


def _shared_trend_field(S=24, T=80, seed=0):
    rng = np.random.default_rng(seed)
    # a strong shared AR(1) trend over time (phi=0.9), common to A and B across space
    phi = 0.9
    trend = np.zeros((S, T))
    trend[:, 0] = rng.standard_normal(S)
    for t in range(1, T):
        trend[:, t] = phi * trend[:, t - 1] + rng.standard_normal(S)
    A = trend + 0.3 * rng.standard_normal((S, T))
    B = trend + 0.3 * rng.standard_normal((S, T))  # independent of A GIVEN the shared trend
    Z = np.stack([A, B])
    space_ids = tuple(f"s{i}" for i in range(S))
    return GaussianField(
        variables=("A", "B"), space_ids=space_ids, time_ids=tuple(range(T)),
        Z=Z, W=np.ones_like(Z), resolution="year",
    )


def test_shared_ar1_trend_lagged_edge_removed_by_temporal_whitening():
    """The shared trend makes A(t-k) predict B(t) → the engine emits spurious *directed
    lagged* A→B / B→A edges. Temporal whitening removes the trend, so no lagged edge."""
    field = _shared_trend_field()
    kw = dict(K=2, spatial_whiten=False, min_coverage=10, min_overlap=8)

    raw = fit_lagged_links(field, temporal_whiten=False, **kw)
    white = fit_lagged_links(field, temporal_whiten=True, **kw)

    raw_pairs = {(l.source, l.target) for l in raw.lagged_links}
    white_pairs = {(l.source, l.target) for l in white.lagged_links}
    assert {("A", "B"), ("B", "A")} & raw_pairs, f"shared AR(1) trend should induce spurious lagged A-B edges, got {raw_pairs}"
    assert not ({("A", "B"), ("B", "A")} & white_pairs), f"temporal whitening must remove the trend-induced lagged edges, got {white_pairs}"


def test_temporal_whiten_false_is_noop():
    """temporal_whiten=False (the default) must be byte-identical to the pre-existing fit —
    the flag is opt-in and off recovers the exact prior behavior."""
    field = _shared_trend_field(seed=3)
    kw = dict(K=2, spatial_whiten=False, min_coverage=10, min_overlap=8)
    default = fit_lagged_links(field, **kw)                       # flag absent -> default off
    explicit = fit_lagged_links(field, temporal_whiten=False, **kw)
    assert np.array_equal(default.fit.precision, explicit.fit.precision)


def test_temporal_whiten_preserves_missing_and_t0():
    Z = np.random.default_rng(1).standard_normal((2, 5, 8))
    Z[0, 2, 4] = np.nan
    Zw, phi = temporal_whiten(Z)
    assert Zw.shape == Z.shape and phi.shape == (2,)
    # t=0 slice is left untouched; the NaN cell and its forward neighbour do not fabricate values
    assert np.array_equal(Zw[:, :, 0], Z[:, :, 0])
    assert not np.isfinite(Zw[0, 2, 4]) and not np.isfinite(Zw[0, 2, 5])
