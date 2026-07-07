"""LDO-MARG-03 — reliability-weighted rank PIT + MI-combined randomized-PIT jitter.

(a) The rank margin's ECDF must be reliability-weighted: a block of low-reliability
outlier cells otherwise distorts every reliable cell's latent rank. Down-weighting them
moves the reliable cells' latent values toward the reliable-only ECDF. All-ones weights
reproduce the unweighted ECDF exactly.

(b) A single frozen auxiliary-uniform draw injects one realization of tie jitter as if it
were signal. Averaging z over independent draws (n_pit_draws>1) shrinks that variance at
tied cells. The default path (weights=None, n_pit_draws=1, use_reliability_ecdf on all-ones
W) stays byte-identical.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.margins import gaussianize_field, randomized_pit_gaussianize


def test_weighted_ecdf_allones_matches_unweighted() -> None:
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)
    x = np.concatenate([rng_a.integers(0, 5, size=200).astype(float), [np.nan] * 20])
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    z_unw = randomized_pit_gaussianize(x, rng=rng_a)
    z_w1 = randomized_pit_gaussianize(x, rng=rng_b, weights=np.ones_like(x))
    m = np.isfinite(z_unw)
    assert np.allclose(z_unw[m], z_w1[m], atol=0, rtol=0)


def test_downweighted_outliers_stop_distorting_reliable_ranks() -> None:
    # 300 reliable cells (moderate spread) + 100 outlier cells far above them.
    rng = np.random.default_rng(11)
    reliable = rng.normal(0.0, 1.0, size=300)
    outliers = rng.normal(20.0, 1.0, size=100)
    x = np.concatenate([reliable, outliers])
    w = np.concatenate([np.ones(300), np.full(100, 1e-4)])  # outliers ~ no reliability
    rel_only = np.arange(300)

    # reliable-only ECDF: the target the weighted margin should approach.
    z_ref = randomized_pit_gaussianize(reliable, rng=np.random.default_rng(1))
    z_unw = randomized_pit_gaussianize(x, rng=np.random.default_rng(1))[rel_only]
    z_wtd = randomized_pit_gaussianize(x, rng=np.random.default_rng(1), weights=w)[rel_only]

    d_unw = float(np.mean(np.abs(z_unw - z_ref)))
    d_wtd = float(np.mean(np.abs(z_wtd - z_ref)))
    # the unweighted ECDF drags reliable cells down (100 phantom "greater" cells);
    # weighting collapses that distortion toward the reliable-only transform.
    assert d_wtd < 0.4 * d_unw
    # sanity: the unweighted ECDF drags reliable cells negative (phantom "greater" mass);
    # weighting recenters them on the reliable-only band (mean ~ 0).
    assert z_unw.mean() < -0.25
    assert abs(z_wtd.mean()) < 0.1


def test_mi_combine_reduces_tie_jitter_variance() -> None:
    # a heavy tie block at zero: the randomized-PIT jitter is pure single-draw noise there.
    x = np.concatenate([np.zeros(120), np.arange(1.0, 61.0)])
    zero = np.arange(120)

    def zero_std(draws: int, seed: int) -> float:
        z = randomized_pit_gaussianize(
            x, rng=np.random.default_rng(seed), n_pit_draws=draws
        )
        return float(np.std(z[zero]))

    s1 = np.mean([zero_std(1, s) for s in range(20)])
    s16 = np.mean([zero_std(16, s) for s in range(20)])
    assert s16 < 0.5 * s1  # ~1/sqrt(draws) shrinkage of the within-tie jitter


def _field(x: np.ndarray, w: np.ndarray) -> LDOField:
    p, S, T = x.shape
    return LDOField(
        variables=tuple(f"v{i}" for i in range(p)),
        space_ids=tuple(f"s{i}" for i in range(S)),
        time_ids=tuple(range(T)),
        X=x,
        W=w,
    )


def test_gaussianize_field_default_byte_identical() -> None:
    rng = np.random.default_rng(5)
    x = rng.integers(0, 6, size=(3, 20, 4)).astype(float)
    x[0, 0, 0] = np.nan
    w = np.ones_like(x)
    f = _field(x, w)

    z_default = gaussianize_field(f, seed=2).Z
    # explicitly-off reliability path == default-on path with all-ones W
    z_off = gaussianize_field(f, seed=2, use_reliability_ecdf=False).Z
    m = np.isfinite(z_default)
    assert np.allclose(z_default[m], z_off[m], atol=0, rtol=0)
