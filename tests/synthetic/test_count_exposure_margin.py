"""LDO-01 — the count-with-exposure copula margin (MSD-III §III.5, denominator principle).

An extensive count's cross-cell variation is dominated by exposure differences. The
count-with-exposure (Poisson-offset) margin models the count *net of* exposure, so the
latent Gaussian carries deviation-from-expected — not a spurious exposure signal that
rank-PIT on raw counts would retain.
"""

from __future__ import annotations

import numpy as np

from pegasus.pirs.ldo.margins import count_exposure_gaussianize, randomized_pit_gaussianize


def _abscorr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return abs(float(np.corrcoef(a[m], b[m])[0, 1]))


def test_count_exposure_margin_removes_exposure_signal() -> None:
    rng = np.random.default_rng(0)
    n = 500
    exposure = rng.uniform(200.0, 8000.0, size=n)     # wildly heterogeneous denominators
    counts = rng.poisson(0.01 * exposure).astype(float)   # same underlying rate everywhere

    z_offset = count_exposure_gaussianize(counts, exposure, rng=np.random.default_rng(1))
    z_rank = randomized_pit_gaussianize(counts, rng=np.random.default_rng(1))

    # rank-PIT of raw counts tracks exposure (more exposure → more events → higher rank)
    assert _abscorr(z_rank, exposure) > 0.5
    # the count-with-exposure margin removes it: latent Z is ~uncorrelated with exposure
    assert _abscorr(z_offset, exposure) < 0.2
    # and the offset margin is a genuine standard-normal transform
    finite = z_offset[np.isfinite(z_offset)]
    assert abs(finite.mean()) < 0.2 and 0.7 < finite.std() < 1.3
