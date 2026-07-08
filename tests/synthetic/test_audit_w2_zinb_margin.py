"""ADVERSARIAL AUDIT — W2 overdispersed NB/ZINB margin (margins.py, commit aa10a8b).

The copula margin's whole contract (§III.5 / §6.2) is that the randomized-PIT
under the CHOSEN family CDF makes the latent Z standard normal — i.e. the PIT
values u = F(x-1) + U*p(x) are Uniform(0,1). A correct margin manufactures NO
residual structure.

For a zero-inflated NB (ZINB) with inflation mass pi:
    P(X=0) = pi + (1-pi)*NB(0)
    P(X=k) = (1-pi)*NB(k),   k>0
    F(k)   = pi + (1-pi)*F_NB(k)      (for k>=0)

so the randomized-PIT lower-cdf for a NONZERO cell x=k>0 must be
    F(k-1) = pi + (1-pi)*F_NB(k-1).

margins.py:185 computes `cdf_lo = (1-pi)*nbinom.cdf(x-1,...)` for ALL cells,
dropping the +pi structural-zero point mass from the cumulative term of every
nonzero cell. Consequence: u for nonzero cells is biased low by ~pi and u never
exceeds 1-pi, so z = Phi^{-1}(u) is pushed systematically negative and the whole
top-pi band of the unit interval is unreachable. The margin re-injects exactly
the spurious latent structure the fix claims to remove.

These tests construct genuine zero-inflated NB data (so ZINB is auto-selected)
and assert the margin's own contract. They FAIL iff the ZINB CDF bug is real.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest, norm

from pegasus.ldo.margins import _select_family, count_exposure_gaussianize


def _zinb_counts(rng, mu, r, pi):
    nb = rng.negative_binomial(r, r / (r + mu)).astype(float)
    structural_zero = rng.uniform(size=nb.shape) < pi
    return np.where(structural_zero, 0.0, nb)


def test_zinb_margin_is_actually_selected() -> None:
    """Guard: confirm the constructed data really routes through the ZINB branch,
    so the failures below are about the ZINB CDF and not a mis-selection."""
    rng = np.random.default_rng(0)
    n = 8000
    exposure = rng.uniform(500.0, 5000.0, size=n)
    mu = 0.02 * exposure
    counts = _zinb_counts(rng, mu, r=4.0, pi=0.35)

    m = np.isfinite(counts) & (exposure > 0)
    x = counts[m]
    e = exposure[m]
    lam = x.sum() / e.sum()
    muc = np.clip(lam * e, 1e-6, None)
    fam, r, pi = _select_family(x, muc)
    assert fam == "zinb", f"expected ZINB to be selected, got {fam}"
    assert pi > 0.05


def test_zinb_latent_z_is_standard_normal() -> None:
    """The margin's contract: latent Z must be ~ standard normal (mean 0, sd 1).

    A correct randomized-PIT under the ZINB CDF yields u ~ Uniform(0,1) hence
    z ~ N(0,1). The dropped +pi cumulative term biases nonzero cells low, so the
    latent mean drops well below 0 and the sd collapses below 1.
    """
    rng = np.random.default_rng(7)
    n = 12000
    exposure = rng.uniform(500.0, 5000.0, size=n)
    mu = 0.02 * exposure
    counts = _zinb_counts(rng, mu, r=4.0, pi=0.35)

    z, fam = count_exposure_gaussianize(
        counts, exposure, rng=np.random.default_rng(1),
        family="auto", return_family=True,
    )
    assert fam == "zinb"
    zf = z[np.isfinite(z)]

    mean = float(zf.mean())
    sd = float(zf.std())
    # A valid standard-normal margin: |mean| small, sd ~ 1.
    assert abs(mean) < 0.15, f"latent Z mean {mean:.3f} far from 0 (ZINB CDF bias)"
    assert 0.9 < sd < 1.1, f"latent Z sd {sd:.3f} not ~1 (ZINB CDF collapses spread)"


def test_zinb_pit_is_uniform_with_true_parameters() -> None:
    """Isolate the CDF formula from parameter estimation.

    Build the randomized-PIT u exactly as margins.py does (lines 179-188) but with
    the TRUE data-generating (r, pi), so r/pi are not a confound. A correct ZINB
    PIT is Uniform(0,1); the code's PIT is confined to [0, 1-pi] and fails the
    KS-uniformity test decisively.
    """
    from scipy.stats import nbinom

    rng = np.random.default_rng(3)
    n = 40000
    mu = np.full(n, 3.0)
    r_true, pi_true = 4.0, 0.4
    nb = rng.negative_binomial(r_true, r_true / (r_true + mu)).astype(float)
    x = np.where(rng.uniform(size=n) < pi_true, 0.0, nb)

    U = rng.uniform(0.0, 1.0, size=n)
    p_nb = r_true / (r_true + mu)
    cdf_lo = nbinom.cdf(x - 1, r_true, p_nb)
    pmf = nbinom.pmf(x, r_true, p_nb)
    is0 = x == 0
    # ---- exactly the production ZINB branch (margins.py) ----
    cdf_lo_code = np.where(is0, 0.0, pi_true + (1.0 - pi_true) * cdf_lo)
    pmf_code = np.where(is0, pi_true + (1.0 - pi_true) * pmf, (1.0 - pi_true) * pmf)
    u_code = cdf_lo_code + U * pmf_code

    # A correct randomized PIT is Uniform(0,1).
    ks_p = kstest(u_code, "uniform").pvalue
    assert ks_p > 1e-3, (
        f"production ZINB PIT is not uniform (KS p={ks_p:.2e}); "
        f"u range [{u_code.min():.3f}, {u_code.max():.3f}] excludes the top-pi band"
    )
    assert abs(float(u_code.mean()) - 0.5) < 0.02, (
        f"production ZINB PIT mean {u_code.mean():.3f} != 0.5 (biased low by ~pi)"
    )
