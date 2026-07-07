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
    """Return (family, r, pi) from Pearson dispersion and excess zeros. r=inf → Poisson."""
    phi = float(np.mean((x - mu) ** 2 / mu))
    if phi <= 1.0:
        return "poisson", np.inf, 0.0
    r = float(np.clip(np.mean(mu) / (phi - 1.0), _R_MIN, _R_MAX))
    p_nb = r / (r + mu)
    zero_obs = float(np.mean(x == 0))
    zero_nb = float(np.mean(p_nb**r))
    if zero_obs > zero_nb + 1e-3 and zero_nb < 1.0 - 1e-9:
        pi = float(np.clip((zero_obs - zero_nb) / (1.0 - zero_nb), 0.0, 1.0 - _EPS))
        return "zinb", r, pi
    return "nb", r, 0.0


def count_exposure_gaussianize(
    counts: np.ndarray,
    exposure: np.ndarray,
    *,
    rng: np.random.Generator,
    family: str = "auto",
    return_family: bool = False,
):
    """Count-with-exposure copula margin (MSD-III §III.5 / MSD-I §6.2).

    ``μ_i = λ·E_i`` (``λ = Σcount/Σexposure``, the log-exposure offset). The margin CDF is
    selected from the data (``family="auto"``): Poisson when equidispersed, NB under
    method-of-moments overdispersion (``r = mean(μ)/(φ-1)``, ``φ`` the Pearson dispersion),
    or ZINB when zeros exceed the NB-implied fraction. Randomized PIT under the chosen CDF
    gives ``u_i = F(x_i-1) + U·p(x_i)``, ``z_i = Φ^{-1}(u_i)`` — so latent ``Z`` carries
    deviation net of exposure without re-manufacturing overdispersion as latent structure.
    Cells with non-positive/absent exposure are left NaN. With ``return_family`` the chosen
    family string is returned alongside the array.
    """
    out = np.full(counts.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(counts) & np.isfinite(exposure) & (exposure > 0)
    chosen = "poisson"
    if mask.sum() < 2:
        return (out, chosen) if return_family else out
    x = counts[mask].astype(np.float64)
    e = exposure[mask].astype(np.float64)
    total_e = e.sum()
    if total_e <= 0:
        return (out, chosen) if return_family else out
    lam = x.sum() / total_e
    mu = np.clip(lam * e, _EPS, None)

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
        p_nb = r / (r + mu)
        cdf_lo = nbinom.cdf(x - 1, r, p_nb)
        pmf = nbinom.pmf(x, r, p_nb)
        if chosen == "zinb":
            # ZINB: inflation mass π added at 0, NB mass scaled by (1-π).
            is0 = x == 0
            cdf_lo = (1.0 - pi) * cdf_lo
            pmf = (1.0 - pi) * pmf
            pmf = np.where(is0, pmf + pi, pmf)
        u = cdf_lo + U * pmf
    out[mask] = ndtri(np.clip(u, _EPS, 1.0 - _EPS))
    return (out, chosen) if return_family else out


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
    for j in range(p):
        x = field.X[j].reshape(-1)
        e = exposure[j].reshape(-1) if exposure is not None else None
        if e is not None and np.isfinite(e).any() and np.nanmax(np.where(np.isfinite(e), e, 0.0)) > 0:
            z, fam = count_exposure_gaussianize(x, e, rng=rng, return_family=True)
            Z[j] = z.reshape(S, T)
            families[field.variables[j]] = fam
        else:
            w = field.W[j].reshape(-1) if use_reliability_ecdf else None
            Z[j] = randomized_pit_gaussianize(
                x, rng=rng, weights=w, n_pit_draws=n_pit_draws
            ).reshape(S, T)
    return GaussianField(
        variables=field.variables,
        space_ids=field.space_ids,
        time_ids=field.time_ids,
        Z=Z,
        W=field.W,
        resolution=field.resolution,
        count_families=families or None,
    )


__all__ = [
    "GaussianField", "gaussianize_field", "randomized_pit_gaussianize",
    "inverse_gaussianize", "count_exposure_gaussianize",
]
