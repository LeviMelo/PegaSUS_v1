"""W2 — overdispersed count margin (MSD-I §6.2).

A pooled equidispersed Poisson margin re-manufactures NB overdispersion as spurious latent
structure: the randomized-PIT Z under the wrong (too-narrow) CDF is over-dispersed, not
standard normal. The auto-selecting NB margin restores near-unit latent variance and picks
the NB family; equidispersed input still selects Poisson (byte-identical fallback).
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.margins import count_exposure_gaussianize


def _nb_counts(rng, mu, r):
    p = r / (r + mu)
    return rng.negative_binomial(r, p).astype(float)


def test_overdispersed_margin_selects_nb_and_calibrates() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    exposure = rng.uniform(500.0, 5000.0, size=n)
    lam, r = 0.01, 3.0                       # constant rate, heavy NB overdispersion
    counts = _nb_counts(rng, lam * exposure, r)

    z_poi, fam_poi = count_exposure_gaussianize(
        counts, exposure, rng=np.random.default_rng(1), family="poisson", return_family=True
    )
    z_nb, fam_auto = count_exposure_gaussianize(
        counts, exposure, rng=np.random.default_rng(1), family="auto", return_family=True
    )

    assert fam_poi == "poisson"
    assert fam_auto == "nb"                   # family selector detects overdispersion

    std_poi = float(np.nanstd(z_poi))
    std_nb = float(np.nanstd(z_nb))
    # the pooled-Poisson margin inflates latent variance (spurious structure);
    # the NB margin restores near-unit variance.
    assert std_poi > 1.3
    assert 0.9 < std_nb < 1.1
    assert std_nb < std_poi
    assert abs(float(np.nanmean(z_nb))) < 0.1


def test_equidispersed_input_selects_poisson() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    exposure = rng.uniform(500.0, 5000.0, size=n)
    counts = rng.poisson(0.01 * exposure).astype(float)   # genuinely equidispersed

    z_auto, fam = count_exposure_gaussianize(
        counts, exposure, rng=np.random.default_rng(3), family="auto", return_family=True
    )
    z_poi = count_exposure_gaussianize(
        counts, exposure, rng=np.random.default_rng(3), family="poisson"
    )

    assert fam == "poisson"                    # no overdispersion → Poisson
    # auto and explicit-Poisson agree byte-for-byte (default fallback is a no-op)
    np.testing.assert_allclose(np.nan_to_num(z_auto), np.nan_to_num(z_poi), atol=1e-12)
    assert 0.9 < float(np.nanstd(z_auto)) < 1.1
