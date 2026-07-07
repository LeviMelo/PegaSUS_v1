"""Spatial BYM varying-coefficient field for an LDO edge (MSD-III §III.3, §III.4(5), §III.7).

An LDO edge ``X → Y`` may be *effect-modified* by place: the relationship is stronger in some
localities than others ("is X→Y stronger where sanitation is poor?"). This estimates the
per-locality slope field ``β_s`` of ``Y ≈ α_s + β_s·X`` with a **BYM/ICAR spatial-smoothness
prior** — the additive GMRF quadratic ``(τ/2)·βᵀ L_W β`` that shrinks neighbouring localities'
coefficients toward each other (distinct from the whitening metric, which removes spatial
nuisance; this is the prior term the LVGLASSO objective lacks). The fitted ``β_s`` surface is
the edge's spatial-heterogeneity field (§III.7 ``spatial_field_ref``).

Because the per-locality data are centered within locality (over time) before the fit, ``β_s``
is the *within-locality* slope, so it is not confounded by between-locality mean differences.
Uninformative localities (no within-locality variation in ``X``) are filled in from their
neighbours through ``L_W`` — the borrow-strength behaviour of a small-area model.

A §III.3 **multiresolution readout** then decomposes ``β_s`` into nested scales —
``national + region + state + muni`` (region = ``cod6[0]``, state = ``cod6[:2]``, muni = full
cod6) — each coarse level a shrunk group effect and the finest level the exact residual, so the
components sum to ``β_s``. The between-scale variance shares say at which resolution the
effect-modification lives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialFieldFit:
    source: str
    target: str
    space_ids: tuple[str, ...]
    beta: np.ndarray                 # (S,) per-locality slope field
    weight: np.ndarray               # (S,) per-locality effective sample weight
    components: dict[str, np.ndarray]  # national/region/state/muni (each (S,), national broadcast)
    heterogeneity: dict[str, float]  # summary: std, range, per-scale variance shares
    tau: float


def _within_locality_stats(X: np.ndarray, Y: np.ndarray, W: np.ndarray):
    """Weighted within-locality (over time) centered sufficient statistics per locality.

    Returns ``(A, D, c)`` with ``A_s = Σ_t w (X−X̄_s)²``, ``D_s = Σ_t w (X−X̄_s)(Y−Ȳ_s)``,
    ``c_s = Σ_t w`` — the normal-equation pieces for the within-locality slope. Cells where
    either variable is missing carry zero weight.
    """
    finite = np.isfinite(X) & np.isfinite(Y)
    w = np.where(finite & (W > 0), W, 0.0)
    Xc = np.where(finite, X, 0.0)
    Yc = np.where(finite, Y, 0.0)
    c = w.sum(axis=1)                                  # (S,)
    safe = np.where(c > 0, c, 1.0)
    xbar = (w * Xc).sum(axis=1) / safe
    ybar = (w * Yc).sum(axis=1) / safe
    dX = Xc - xbar[:, None]
    dY = Yc - ybar[:, None]
    A = (w * dX * dX).sum(axis=1)                      # (S,)
    D = (w * dX * dY).sum(axis=1)                      # (S,)
    return A, D, c


def fit_spatial_varying_coefficient(
    field, source: str, target: str, *,
    tau: float = 1.0, ridge: float = 1e-6, kappa: float = 0.0, laplacian=None,
) -> SpatialFieldFit | None:
    """Estimate the BYM-smoothed per-locality slope field ``β_s`` for the edge ``source→target``.

    Solves the GMRF-penalized normal equations ``(diag(A) + τ·L_W + ridge·I) β = D`` — a single
    sparse SPD solve. ``L_W`` is the municipality structural-adjacency Laplacian
    (:func:`build_spatial_precision_sparse` at ``κ=0``), or a caller-supplied ``laplacian`` (any
    ``scipy.sparse`` SPD/graph-Laplacian). Returns ``None`` if no locality is informative
    (no within-locality ``X`` variation anywhere), so the caller simply omits the field.
    """
    variables = list(field.variables)
    if source not in variables or target not in variables:
        return None
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    si, ti = variables.index(source), variables.index(target)
    X, Y = field.Z[si], field.Z[ti]                    # (S, T)
    W = field.W[si] if getattr(field, "W", None) is not None else np.ones_like(X)
    A, D, c = _within_locality_stats(X, Y, W)
    if not np.any(A > 0):
        return None
    S = len(field.space_ids)

    if laplacian is not None:
        L_W = laplacian.tocsr() if sp.issparse(laplacian) else sp.csr_matrix(laplacian)
    else:
        from pegasus.ldo.precision import build_spatial_precision_sparse
        L_W = build_spatial_precision_sparse(field.space_ids, kappa=float(kappa))  # κ=0 → L_W=D−A

    Aop = sp.diags(A + ridge) + float(tau) * L_W
    beta = np.asarray(spsolve(Aop.tocsc(), D), dtype=np.float64)

    components = _multiresolution_decompose(beta, field.space_ids, c)
    heterogeneity = _heterogeneity_summary(beta, c, components)
    return SpatialFieldFit(
        source=source, target=target, space_ids=tuple(field.space_ids),
        beta=beta, weight=c, components=components, heterogeneity=heterogeneity, tau=float(tau),
    )


def _weighted_group_mean(values: np.ndarray, weights: np.ndarray, labels: np.ndarray,
                         *, pseudo: float = 1.0) -> np.ndarray:
    """Per-element shrunk weighted group mean: broadcast ``(n_g/(n_g+pseudo))·mean_g`` back.

    ``pseudo`` is the shrinkage pseudo-count (empirical-Bayes toward 0): small groups are pulled
    toward the global level, large groups keep their own mean — the coarse→fine borrowing §III.3
    prescribes.
    """
    out = np.zeros_like(values, dtype=np.float64)
    for lab in np.unique(labels):
        m = labels == lab
        wg = weights[m]
        wsum = wg.sum()
        if wsum <= 0:
            continue
        mean_g = float((values[m] * wg).sum() / wsum)
        shrink = wsum / (wsum + pseudo)
        out[m] = shrink * mean_g
    return out


def _multiresolution_decompose(beta: np.ndarray, space_ids: tuple[str, ...],
                               weight: np.ndarray) -> dict[str, np.ndarray]:
    """§III.3 nested-scale decomposition ``β = national + region + state + muni`` (sum-exact).

    ``region = cod6[0]`` (IBGE macroregion), ``state = cod6[:2]`` (UF), ``muni`` = full cod6.
    National is the weighted mean; region/state are shrunk group means of the running residual;
    muni is the exact remaining residual so the four components always sum to ``β``.
    """
    ids = [str(s) for s in space_ids]
    region = np.array([s[0] if len(s) >= 1 else "0" for s in ids])
    state = np.array([s[:2] if len(s) >= 2 else s for s in ids])
    wsum = weight.sum()
    national_val = float((beta * weight).sum() / wsum) if wsum > 0 else float(beta.mean())
    national = np.full_like(beta, national_val)
    r1 = beta - national
    region_c = _weighted_group_mean(r1, weight, region)
    r2 = r1 - region_c
    state_c = _weighted_group_mean(r2, weight, state)
    muni_c = r2 - state_c                               # exact residual → components sum to β
    return {"national": national, "region": region_c, "state": state_c, "muni": muni_c}


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    wsum = w.sum()
    if wsum <= 0:
        return float(np.var(x))
    mean = (x * w).sum() / wsum
    return float(((x - mean) ** 2 * w).sum() / wsum)


def _heterogeneity_summary(beta: np.ndarray, weight: np.ndarray,
                           components: dict[str, np.ndarray]) -> dict[str, float]:
    """Effect-modification magnitude + which scale it lives at (variance shares)."""
    total_var = _weighted_var(beta, weight)
    shares = {}
    for scale in ("region", "state", "muni"):
        shares[f"{scale}_var"] = _weighted_var(components[scale], weight)
    denom = sum(shares.values()) or 1.0
    out = {
        "beta_mean": float((beta * weight).sum() / (weight.sum() or 1.0)),
        "beta_std": float(np.sqrt(max(total_var, 0.0))),
        "beta_min": float(beta.min()),
        "beta_max": float(beta.max()),
        "total_var": total_var,
    }
    for scale in ("region", "state", "muni"):
        out[f"{scale}_var_share"] = float(shares[f"{scale}_var"] / denom)
    return out


def write_spatial_field(fit: SpatialFieldFit, path) -> str:
    """Persist the ``β_s`` field + multiresolution components to a parquet sidecar; return path."""
    import polars as pl

    df = pl.DataFrame({
        "space_id": list(fit.space_ids),
        "beta": fit.beta,
        "weight": fit.weight,
        "comp_national": fit.components["national"],
        "comp_region": fit.components["region"],
        "comp_state": fit.components["state"],
        "comp_muni": fit.components["muni"],
    })
    df.write_parquet(str(path), compression="zstd")
    return str(path)


__all__ = [
    "SpatialFieldFit",
    "fit_spatial_varying_coefficient",
    "write_spatial_field",
]
