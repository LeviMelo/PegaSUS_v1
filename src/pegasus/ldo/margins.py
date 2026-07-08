"""LDO Layer 0 — copula margins (MSD-II §II.6.1, MII-LDO-01).

Push each variable to a latent Gaussian scale via its marginal CDF:
``Z_{j,s,t} = Φ^{-1}(F_j(X_{j,s,t}))``. ``F_j`` is the admissible family CDF
(§6.2 / ``compute/glm.py``) when declared; otherwise — and always for sparse,
zero-inflated counts — the **randomized probability integral transform** (the
empirical/rank copula margin, the discrete-data-correct form used by Dunn–Smyth
randomized-quantile residuals). Reliability weights ``W`` propagate unchanged.

The randomized PIT maps ties (e.g. the many zero cells of a rare-event count)
uniformly across their CDF band, so the latent ``Z`` is genuinely continuous
Gaussian rather than collapsing all zeros onto one value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import nbinom, poisson

from pegasus.ldo.assemble import LDOField

_EPS = 1e-6
_R_MIN, _R_MAX = 1e-2, 1e6


@dataclass
class GaussianField:
    variables: tuple[str, ...]
    space_ids: tuple[str, ...]
    time_ids: tuple[int, ...]
    Z: np.ndarray   # (p, S, T) latent Gaussian, NaN where unobserved
    W: np.ndarray   # (p, S, T) reliability weights (propagated from assembly)
    resolution: str = "year"
    count_families: dict[str, str] | None = None   # var → chosen count family (provenance)
    margin_calibration: dict[str, float] | None = None  # var → PIT-uniformity KS p-value (count-exposure)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.Z.shape


def randomized_pit_gaussianize(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
    n_pit_draws: int = 1,
) -> np.ndarray:
    """Randomized-PIT → Gaussian for one flattened variable (NaN preserved).

    For each observed value ``x``: ``u = (#{obs < x} + U·#{obs = x}) / n`` with
    ``U ~ Uniform(0,1)``, then ``z = Φ^{-1}(u)``. Continuous data (no ties) reduces
    to the usual rank PIT; discrete/zero-inflated data spreads ties across their
    band.

    ``weights`` (per-cell reliability, same shape) uses a weighted ECDF: the less-than
    and tie masses are weight sums over total weight, so a low-reliability cell no
    longer distorts every other cell's rank. ``weights=None`` reproduces the unweighted
    ECDF exactly. ``n_pit_draws>1`` averages ``z`` over independent ``U`` draws
    (Rubin-combines the randomized-quantile jitter), stabilizing ties.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    obs = values[mask]
    n = obs.size
    if n < 2:
        return out
    order_idx = np.argsort(obs, kind="stable")
    order = obs[order_idx]
    if weights is None:
        less = np.searchsorted(order, obs, side="left").astype(np.float64)
        leq = np.searchsorted(order, obs, side="right").astype(np.float64)
        eq = np.maximum(leq - less, 1.0)
        denom = float(n)
    else:
        w = np.asarray(weights, dtype=np.float64)[mask]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        wsorted = w[order_idx]
        cum = np.concatenate([[0.0], np.cumsum(wsorted)])
        lo = np.searchsorted(order, obs, side="left")
        hi = np.searchsorted(order, obs, side="right")
        less = cum[lo]
        leq = cum[hi]
        eq = np.maximum(leq - less, w)  # own-weight is the minimal tie mass for a positive-weight cell
        denom = float(cum[-1])
    if denom <= 0:
        return out
    z_acc = np.zeros(n, dtype=np.float64)
    for _ in range(max(1, int(n_pit_draws))):
        u = (less + rng.uniform(0.0, 1.0, size=n) * eq) / denom
        z_acc += ndtri(np.clip(u, _EPS, 1.0 - _EPS))
    out[mask] = z_acc / max(1, int(n_pit_draws))
    return out


def inverse_gaussianize(z: np.ndarray, observed_values: np.ndarray) -> np.ndarray:
    """§III.5 'mapped back': the inverse copula map ``F_j⁻¹(Φ(z))`` from the latent Gaussian scale
    to a variable's NATIVE scale, via the empirical quantile function of its observed values.

    Completes the round trip (forward :func:`randomized_pit_gaussianize` models dependence on the
    Gaussian scale; this returns a modelled/latent quantile to native units so a readout is
    interpretable). Note the LDO's emitted edge weights are *partial correlations* — unitless and
    scale-invariant by construction, so they need no back-map — but any native-scale readout (a
    predicted count, a counterfactual level) uses this inverse. NaN where no observations exist.
    """
    from scipy.special import ndtr

    obs = np.asarray(observed_values, dtype=np.float64)
    obs = np.sort(obs[np.isfinite(obs)])
    z = np.asarray(z, dtype=np.float64)
    if obs.size == 0:
        return np.full(z.shape, np.nan)
    u = np.clip(ndtr(z), _EPS, 1.0 - _EPS)
    idx = np.clip((u * obs.size).astype(int), 0, obs.size - 1)
    return obs[idx]


def _select_family(x: np.ndarray, mu: np.ndarray) -> tuple[str, float, float]:
    """Return (family, r, pi) from Pearson dispersion and excess zeros. r=inf → Poisson.

    ``mu`` is the MARGINAL expected count (from the exposure offset ``lam=Σx/ΣE``). Under
    zero-inflation the offset estimates ``(1−π)·λ_nb``, so the NB-component mean is ``μ/(1−π)``
    and the Pearson dispersion is inflated by the excess zeros. For ZINB we fit (π, r) jointly by
    fixed-point iteration: r from the NB-component dispersion at ``μ/(1−π)``, π so the model zero
    fraction ``π+(1−π)·NB₀`` matches observed. The zinb caller uses ``μ/(1−π)`` in the NB CDF."""
    phi = float(np.mean((x - mu) ** 2 / mu))
    if phi <= 1.0:
        return "poisson", np.inf, 0.0
    r = float(np.clip(np.mean(mu) / (phi - 1.0), _R_MIN, _R_MAX))
    zero_obs = float(np.mean(x == 0))
    zero_nb = float(np.mean((r / (r + mu)) ** r))
    if not (zero_obs > zero_nb + 1e-3 and zero_nb < 1.0 - 1e-9):
        return "nb", r, 0.0
    pi = float(np.clip((zero_obs - zero_nb) / (1.0 - zero_nb), 0.0, 0.95))
    for _ in range(100):
        mu_nb = mu / (1.0 - pi)
        var = float(np.mean((x - (1.0 - pi) * mu_nb) ** 2))          # marginal variance about the ZINB mean
        mbar = float(np.mean(mu_nb))
        phi_nb = var / max((1.0 - pi) * mbar, _EPS)
        r = float(np.clip(mbar / max(phi_nb - 1.0, 1e-3), _R_MIN, _R_MAX))
        nb0 = float(np.mean((r / (r + mu_nb)) ** r))
        new_pi = float(np.clip((zero_obs - nb0) / (1.0 - nb0), 0.0, 0.95)) if nb0 < 1.0 - 1e-9 else pi
        if abs(new_pi - pi) < 1e-7:
            pi = new_pi
            break
        pi = new_pi
    return "zinb", r, pi


def _per_muni_eb_rate(counts: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Per-municipality empirical-Bayes rate ``λ_s`` (Poisson-Gamma), shrinking each muni's raw
    rate toward the global ``λ`` by the between-muni dispersion (§LDO-MARGIN-10 varying baseline /
    the within-estimator / municipality fixed effect on the rate). ``counts``/``exposure`` are
    ``(S,T)``; returns ``λ_s`` of shape ``(S,)``.

    MoM Gamma prior (mean ``λ``, variance ``σ²`` = exposure-weighted between-muni rate variance minus
    the mean Poisson sampling variance): posterior mean ``λ_s=(Σ_t x + α)/(Σ_t E + β)``, ``β=λ/σ²``,
    ``α=λβ``. So ``σ²→0`` (no genuine between-muni dispersion) ⇒ ``λ_s→λ`` (pooled — the prior
    behaviour, no drift on a homogeneous field); large ``σ²`` ⇒ ``λ_s→`` the raw per-muni rate (full
    fixed effects). This makes the latent ``Z`` a deviation from the muni's OWN expected count — net
    of the time-invariant spatial-rate surface that pooled-λ baked in as spurious dependence."""
    S = counts.shape[0]
    obs = np.isfinite(counts) & np.isfinite(exposure) & (exposure > 0)
    x_s = np.where(obs, counts, 0.0).sum(axis=1)
    e_s = np.where(obs, exposure, 0.0).sum(axis=1)
    tot_e = float(e_s.sum())
    lam = x_s.sum() / tot_e if tot_e > 0 else 0.0
    valid = e_s > 0
    if lam <= 0.0 or int(valid.sum()) < 3:
        return np.full(S, lam, dtype=np.float64)
    w = e_s[valid]
    rv = x_s[valid] / np.maximum(e_s[valid], _EPS)
    wmean = float((w * rv).sum() / w.sum())
    v_obs = float((w * (rv - wmean) ** 2).sum() / w.sum())      # exposure-weighted observed rate var
    v_samp = float(np.mean(lam / np.maximum(w, _EPS)))          # mean Poisson sampling var
    sigma2 = max(v_obs - v_samp, 0.0)
    if sigma2 <= 1e-18:
        return np.full(S, lam, dtype=np.float64)                # no between-muni dispersion → pooled
    beta = lam / sigma2
    alpha = lam * beta
    lam_s = (x_s + alpha) / (e_s + beta)
    return np.where(valid, lam_s, lam)


def count_exposure_gaussianize(
    counts: np.ndarray,
    exposure: np.ndarray,
    *,
    rng: np.random.Generator,
    family: str = "auto",
    space_baseline: bool = True,
    return_family: bool = False,
    return_calibration: bool = False,
):
    """Count-with-exposure copula margin (MSD-III §III.5 / MSD-I §6.2).

    The expected count is ``μ_{s,t} = λ_s·E_{s,t}``. With ``space_baseline`` (default) and a 2-D
    ``(S,T)`` input, ``λ_s`` is the per-municipality empirical-Bayes rate (:func:`_per_muni_eb_rate`)
    — a municipality fixed effect on the rate, so latent ``Z`` is deviation from the muni's OWN
    expected count, NOT the national average (LDO-MARGIN-10: a single pooled ``λ`` bakes the
    time-invariant spatial-rate surface into ``Z`` as a spurious dependence the linear whitener
    cannot fully remove). On a homogeneous field, or 1-D input, or ``space_baseline=False``, ``λ_s``
    collapses to the pooled ``λ = Σcount/Σexposure`` (the prior behaviour).

    The margin CDF is data-selected (``family="auto"``): Poisson when equidispersed, NB under
    method-of-moments overdispersion, ZINB when zeros exceed the NB-implied fraction. Randomized PIT
    ``u = F(x-1) + U·p(x)``, ``z = Φ^{-1}(u)``. With ``return_calibration`` a KS-uniformity p-value of
    ``u`` (a goodness-of-fit gate — a mis-calibrated margin yields non-uniform ``u``) is also returned.
    Cells with non-positive/absent exposure are left NaN."""
    counts = np.asarray(counts, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(counts) & np.isfinite(exposure) & (exposure > 0)
    chosen, ks_p = "poisson", 1.0

    def _ret(o):
        r = (o,)
        if return_family:
            r = r + (chosen,)
        if return_calibration:
            r = r + (ks_p,)
        return r[0] if len(r) == 1 else r

    if mask.sum() < 2:
        return _ret(out)
    # per-cell expected count μ over the full grid, then restrict to observed cells for the PIT
    two_d = counts.ndim == 2 and counts.shape == exposure.shape and counts.shape[0] >= 3
    if space_baseline and two_d:
        lam_s = _per_muni_eb_rate(counts, exposure)          # (S,) municipality fixed-effect rate
        mu_grid = lam_s[:, None] * exposure
    else:
        total_e = float(exposure[mask].sum())
        if total_e <= 0:
            return _ret(out)
        mu_grid = (counts[mask].sum() / total_e) * exposure  # pooled λ (homogeneous / 1-D fallback)
    x = counts[mask]
    mu = np.clip(mu_grid[mask], _EPS, None)

    chosen = family
    r, pi = np.inf, 0.0
    if family == "auto":
        chosen, r, pi = _select_family(x, mu)
    elif family in ("nb", "zinb"):
        _, r, pi = _select_family(x, mu)
        r = r if np.isfinite(r) else _R_MAX

    U = rng.uniform(0.0, 1.0, size=x.size)
    if chosen == "poisson":
        u = poisson.cdf(x - 1, mu) + U * poisson.pmf(x, mu)
    else:
        # ZINB: the NB component's mean is μ/(1−π) (the offset estimates the marginal (1−π)·λ_nb).
        mu_c = mu / (1.0 - pi) if chosen == "zinb" else mu
        p_nb = r / (r + mu_c)
        cdf_lo = nbinom.cdf(x - 1, r, p_nb)
        pmf = nbinom.pmf(x, r, p_nb)
        if chosen == "zinb":
            # ZINB CDF F(k)=π·[k≥0]+(1-π)·F_NB(k). Randomized-PIT lower cdf F(k-1):
            # k==0 → 0; k≥1 → π+(1-π)·F_NB(k-1). pmf: p0=π+(1-π)·NB(0), else (1-π)·NB(k).
            is0 = x == 0
            cdf_lo = np.where(is0, 0.0, pi + (1.0 - pi) * cdf_lo)
            pmf = np.where(is0, pi + (1.0 - pi) * pmf, (1.0 - pi) * pmf)
        u = cdf_lo + U * pmf
    u = np.clip(u, _EPS, 1.0 - _EPS)
    # §LDO-MARGIN-10 PIT-uniformity gate: under a correct margin u~Uniform(0,1); a mis-specified
    # family/baseline piles u near 0/1. The KS p-value flags a mis-calibrated margin so the caller
    # can downgrade that variable's edges rather than trust a distorted latent Z. (n>8 to be meaningful.)
    if return_calibration and x.size > 8:
        from scipy.stats import kstest
        ks_p = float(kstest(u, "uniform").pvalue)
    out[mask] = ndtri(u)
    return _ret(out)


def gaussianize_field(
    field: LDOField,
    *,
    seed: int = 0,
    exposure: np.ndarray | None = None,
    use_reliability_ecdf: bool = True,
    n_pit_draws: int = 1,
) -> GaussianField:
    """Apply the copula margin to every variable of an assembled LDO field.

    ``exposure`` (optional ``(p, S, T)``) supplies a per-variable denominator; a variable
    with positive exposure uses the overdispersion-aware count-with-exposure margin
    (Poisson/NB/ZINB auto-selected, §III.5 / §6.2), the rest use the randomized-PIT rank
    margin. The chosen count family per variable is recorded on ``count_families``.

    ``use_reliability_ecdf`` weights the rank margin by ``field.W`` (all-ones ⇒ prior
    behavior); ``n_pit_draws`` averages the randomized-PIT jitter over independent draws.
    """
    rng = np.random.default_rng(seed)
    p, S, T = field.shape
    Z = np.full((p, S, T), np.nan, dtype=np.float64)
    families: dict[str, str] = {}
    calibration: dict[str, float] = {}
    for j in range(p):
        e2 = exposure[j] if exposure is not None else None  # (S,T), kept 2-D for the per-muni baseline
        if e2 is not None and np.isfinite(e2).any() and np.nanmax(np.where(np.isfinite(e2), e2, 0.0)) > 0:
            z, fam, ksp = count_exposure_gaussianize(
                field.X[j], e2, rng=rng, return_family=True, return_calibration=True
            )
            Z[j] = z  # already (S,T)
            families[field.variables[j]] = fam
            calibration[field.variables[j]] = ksp
        else:
            w = field.W[j].reshape(-1) if use_reliability_ecdf else None
            Z[j] = randomized_pit_gaussianize(
                field.X[j].reshape(-1), rng=rng, weights=w, n_pit_draws=n_pit_draws
            ).reshape(S, T)
    return GaussianField(
        variables=field.variables,
        space_ids=field.space_ids,
        time_ids=field.time_ids,
        Z=Z,
        W=field.W,
        resolution=field.resolution,
        count_families=families or None,
        margin_calibration=calibration or None,
    )


__all__ = [
    "GaussianField", "gaussianize_field", "randomized_pit_gaussianize",
    "inverse_gaussianize", "count_exposure_gaussianize",
]
