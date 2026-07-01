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

from pegasus.pirs.ldo.assemble import LDOField

_EPS = 1e-6


@dataclass
class GaussianField:
    variables: tuple[str, ...]
    space_ids: tuple[str, ...]
    time_ids: tuple[int, ...]
    Z: np.ndarray   # (p, S, T) latent Gaussian, NaN where unobserved
    W: np.ndarray   # (p, S, T) reliability weights (propagated from assembly)
    resolution: str = "year"

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.Z.shape


def randomized_pit_gaussianize(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Randomized-PIT → Gaussian for one flattened variable (NaN preserved).

    For each observed value ``x``: ``u = (#{obs < x} + U·#{obs = x}) / n`` with
    ``U ~ Uniform(0,1)``, then ``z = Φ^{-1}(u)``. Continuous data (no ties) reduces
    to the usual rank PIT; discrete/zero-inflated data spreads ties across their
    band.
    """
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    obs = values[mask]
    n = obs.size
    if n < 2:
        return out
    order = np.sort(obs)
    less = np.searchsorted(order, obs, side="left").astype(np.float64)
    leq = np.searchsorted(order, obs, side="right").astype(np.float64)
    eq = np.maximum(leq - less, 1.0)
    u = (less + rng.uniform(0.0, 1.0, size=n) * eq) / n
    u = np.clip(u, _EPS, 1.0 - _EPS)
    out[mask] = ndtri(u)
    return out


def gaussianize_field(field: LDOField, *, seed: int = 0) -> GaussianField:
    """Apply the copula margin to every variable of an assembled LDO field."""
    rng = np.random.default_rng(seed)
    p, S, T = field.shape
    Z = np.full((p, S, T), np.nan, dtype=np.float64)
    for j in range(p):
        Z[j] = randomized_pit_gaussianize(field.X[j].reshape(-1), rng=rng).reshape(S, T)
    return GaussianField(
        variables=field.variables,
        space_ids=field.space_ids,
        time_ids=field.time_ids,
        Z=Z,
        W=field.W,
        resolution=field.resolution,
    )


__all__ = ["GaussianField", "gaussianize_field", "randomized_pit_gaussianize"]
