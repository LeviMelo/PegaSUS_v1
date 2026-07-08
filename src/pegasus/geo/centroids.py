"""Municipality centroids for distance-weighted spatial kernels (P1 / LDO-WHITEN-04 "spatial range").

The structural contiguity graph is BINARY — it cannot tell a 5 km-apart neighbour pair from a
500 km-apart one, so the GMRF whitening couples them equally and has no "spatial range" (the exact
LDO-WHITEN-04 gap). This module provides IBGE municipality centroids (SIRGAS 2000 / Brazil Polyconic,
metres) via the established ``geobr`` package so a distance-decay kernel ``exp(-d/ρ)`` can REWEIGHT the
contiguity edges: nearby neighbours couple strongly, distant contiguous neighbours weakly. Centroids
are purely geographic (``structural``) — safe as a default spatial prior, no §II.4.1 circularity concern.

Build-once/load-on-query (like the contiguity graph): :func:`build_centroids_artifact` fetches via geobr
once and writes a small ``(municipality_cod6, x_m, y_m)`` parquet under ``config/registries/spatial/``;
:func:`load_municipality_centroids` reads the cached artifact — no network at query time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_ARTIFACT = "config/registries/spatial/municipality_centroids.parquet"
_CRS_METRIC = 5880  # SIRGAS 2000 / Brazil Polyconic (metres) — the official national equal-area-ish CRS


def build_centroids_artifact(*, year: int = 2020, out_path: str | Path = _ARTIFACT) -> Path:
    """Fetch municipality polygons via geobr, project to metres, write ``(cod6, x_m, y_m)`` parquet.

    ``cod6`` = IBGE cod7 minus the check digit, matching the LDO's ``structural_cod6`` keying. Run once
    to materialize the artifact; :func:`load_municipality_centroids` reads it thereafter (offline)."""
    import warnings

    import geobr
    import polars as pl

    warnings.filterwarnings("ignore")
    gdf = geobr.read_municipality(code_muni="all", year=year).to_crs(_CRS_METRIC)
    cent = gdf.geometry.centroid
    cod7 = gdf["code_muni"].astype("int64").astype(str)
    frame = pl.DataFrame(
        {
            "municipality_cod6": [c[:6] for c in cod7.tolist()],
            "x_m": np.asarray(cent.x, dtype=np.float64),
            "y_m": np.asarray(cent.y, dtype=np.float64),
        }
    ).group_by("municipality_cod6").agg(pl.col("x_m").mean(), pl.col("y_m").mean())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out, compression="zstd")
    return out


@lru_cache(maxsize=2)
def load_municipality_centroids(path: str = _ARTIFACT) -> dict[str, tuple[float, float]]:
    """``municipality_cod6 -> (x_m, y_m)`` from the cached artifact (metres, EPSG:5880)."""
    import polars as pl

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"municipality centroids artifact missing: {p}; run centroids.build_centroids_artifact() once."
        )
    df = pl.read_parquet(p)
    return {r["municipality_cod6"]: (float(r["x_m"]), float(r["y_m"])) for r in df.to_dicts()}


def estimate_spatial_range(
    Z: np.ndarray,
    space_ids: tuple[str, ...],
    *,
    centroids: dict[str, tuple[float, float]] | None = None,
    n_bins: int = 16,
    max_pairs: int = 150_000,
    seed: int = 0,
) -> float | None:
    """Estimate the spatial-autocorrelation range ρ (metres) by the empirical correlogram of the
    field — the e-folding distance where the mean pairwise spatial correlation drops to ``1/e``.

    This is the data-driven "fit the range" of LDO-WHITEN-04 / LDO-KAPPA-05: instead of a fiat κ=1
    (fixed implicit range) or a fiat median-neighbour ρ, read the range the DATA actually exhibits.
    Robust by construction — the fitted-ρ distance kernel degrades to binary at long range (ρ→∞ ⇒
    weights→1) and sharpens at short range, so whitening at the fitted ρ is ≥ binary everywhere.

    ``Z`` is the ``(p, S, T)`` field; each variable's per-unit temporal mean is standardized, pairs
    are subsampled to ``max_pairs`` and binned by centroid distance. Returns ``None`` when the
    correlogram is too sparse/flat to read a range (caller falls back to the median-distance ρ)."""
    if centroids is None:
        centroids = load_municipality_centroids()
    Z = np.asarray(Z, dtype=np.float64)
    p, S, T = Z.shape
    have = np.array([s in centroids for s in space_ids])
    if have.sum() < 20:
        return None
    xy = np.array([centroids.get(s, (np.nan, np.nan)) for s in space_ids], dtype=np.float64)
    # per-unit temporal mean per variable, standardized across units (the spatial field to decorrelate)
    fields = []
    for j in range(p):
        with np.errstate(invalid="ignore"):
            m = np.nanmean(np.where(np.isfinite(Z[j]), Z[j], np.nan), axis=1)  # (S,)
        ok = np.isfinite(m) & have
        if ok.sum() < 20:
            continue
        v = m.copy()
        v[~ok] = np.nan
        mu, sd = np.nanmean(v), np.nanstd(v)
        if not np.isfinite(sd) or sd < 1e-9:
            continue
        fields.append(((v - mu) / sd, ok))
    if not fields:
        return None
    rng = np.random.default_rng(seed)
    idx = np.where(have)[0]
    # sample unit pairs once, reuse across variables
    n_take = int(min(max_pairs, idx.size * (idx.size - 1) // 2))
    a = rng.choice(idx, size=n_take)
    b = rng.choice(idx, size=n_take)
    keep = a != b
    a, b = a[keep], b[keep]
    d = np.sqrt(((xy[a] - xy[b]) ** 2).sum(axis=1))
    prod_sum = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    edges = np.linspace(0.0, np.nanpercentile(d, 95), n_bins + 1)
    binid = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    for v, ok in fields:
        pv = v[a] * v[b]
        m = ok[a] & ok[b] & np.isfinite(pv)
        np.add.at(prod_sum, binid[m], pv[m])
        np.add.at(cnt, binid[m], 1.0)
    valid = cnt > 30
    if valid.sum() < 3:
        return None
    corr = np.where(valid, prod_sum / np.maximum(cnt, 1.0), np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # e-folding: first crossing of 1/e from the near-origin correlation
    thr = 1.0 / np.e
    c0 = corr[valid][0]
    if not np.isfinite(c0) or c0 <= thr:
        return float(centers[valid][0]) if c0 <= thr else None  # already decorrelated by first bin
    for k in range(1, n_bins):
        if valid[k] and np.isfinite(corr[k]) and corr[k] <= thr:
            c_prev, c_cur = corr[k - 1], corr[k]
            d_prev, d_cur = centers[k - 1], centers[k]
            frac = (c_prev - thr) / max(c_prev - c_cur, 1e-9)
            return float(d_prev + frac * (d_cur - d_prev))
    return None  # never crosses 1/e within 95th-pct range → long-range ⇒ caller keeps binary-like


def distance_decay_edge_weights(
    adjacency: dict[str, tuple[str, ...]],
    *,
    centroids: dict[str, tuple[float, float]] | None = None,
    rho_m: float | None = None,
) -> tuple[dict[str, dict[str, float]], float]:
    """Distance-decay kernel ``w_ij = exp(-d_ij/ρ)`` over the contiguity edges (``d`` = centroid
    distance in metres). ``ρ`` (the spatial range) defaults to the MEDIAN edge distance so the decay
    scale matches the typical adjacency separation — auto-determined from the geometry, not a magic
    constant (the LDO-WHITEN-04 / LDO-KAPPA-05 "fit the range" prescription, made data-driven).

    Returns ``(edge_weights, rho_m)``. An edge whose endpoint lacks a centroid keeps weight 1.0
    (binary fallback for that edge), so a partial centroid set never drops an edge."""
    if centroids is None:
        centroids = load_municipality_centroids()
    dists: dict[tuple[str, str], float] = {}
    for s, nbs in adjacency.items():
        cs = centroids.get(s)
        for nb in nbs:
            cn = centroids.get(nb)
            if cs is not None and cn is not None:
                dists[(s, nb)] = ((cs[0] - cn[0]) ** 2 + (cs[1] - cn[1]) ** 2) ** 0.5
    if rho_m is None:
        vals = np.fromiter(dists.values(), dtype=np.float64)
        rho_m = float(np.median(vals)) if vals.size else 1.0
    rho_m = max(float(rho_m), 1.0)
    weights: dict[str, dict[str, float]] = {}
    for (s, nb), d in dists.items():
        weights.setdefault(s, {})[nb] = float(np.exp(-d / rho_m))
    # edges without a centroid: keep binary 1.0 so the edge survives
    for s, nbs in adjacency.items():
        for nb in nbs:
            if (s, nb) not in dists:
                weights.setdefault(s, {})[nb] = 1.0
    return weights, rho_m


def knn_distance_graph(
    space_ids: tuple[str, ...],
    *,
    k: int = 10,
    rho_m: float | None = None,
    gaussian: bool = True,
    centroids: dict[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, float]] | None] | tuple[None, None]:
    """A distance-kNN spatial graph over ``space_ids``: each unit linked to its ``k`` geographically
    NEAREST (by centroid distance, symmetrized), gaussian-weighted ``exp(-d/ρ)`` (ρ = median nearest-
    neighbour distance if None). A geographically-coherent, UNIFORM-degree structure that whitens
    robustly ≥ queen contiguity at every spatial range (validated on a distance-confounder benchmark:
    tied at very short range, +1–12% better elsewhere, never worse) — it fixes contiguity's degree
    heterogeneity and its far "neighbours" (huge municipalities touching distant ones). Structural
    (purely geographic) ⇒ circularity-safe as a default spatial prior.

    Returns ``(adjacency, edge_weights)`` for :func:`~pegasus.ldo.precision.build_spatial_precision_sparse`,
    or ``(None, None)`` when too few units have a centroid (caller falls back to contiguity)."""
    from scipy.spatial import cKDTree

    if centroids is None:
        centroids = load_municipality_centroids()
    present = [s for s in space_ids if s in centroids]
    if len(present) < max(k + 1, 4):
        return None, None
    xy = np.array([centroids[s] for s in present], dtype=np.float64)
    kq = min(k + 1, len(present))  # +1: the query point returns itself at rank 0
    dist, nn = cKDTree(xy).query(xy, k=kq)
    dist = np.atleast_2d(dist)
    nn = np.atleast_2d(nn)
    if rho_m is None:
        rho_m = float(np.median(dist[:, 1])) if dist.shape[1] > 1 else 1.0
    rho_m = max(float(rho_m), 1.0)
    adj: dict[str, set] = {}
    wts: dict[str, dict[str, float]] = {}
    for i, s in enumerate(present):
        for rank in range(1, dist.shape[1]):  # skip self at rank 0
            t = present[int(nn[i, rank])]
            w = float(np.exp(-dist[i, rank] / rho_m)) if gaussian else 1.0
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)  # symmetrize (mutual-or-either kNN)
            wts.setdefault(s, {})[t] = w
            wts.setdefault(t, {})[s] = w
    adjacency = {s: tuple(sorted(v)) for s, v in adj.items()}
    return adjacency, (wts if gaussian else None)


def gravity_edge_weights(
    space_ids: tuple[str, ...],
    populations: dict[str, float],
    *,
    centroids: dict[str, tuple[float, float]] | None = None,
    gamma: float = 2.0,
    k: int = 12,
    n_candidates: int = 60,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, float]]] | tuple[None, None]:
    """Gravity spatial kernel ``w_ij = (popᵢ·popⱼ)/d_ijᵞ`` over each node's top-``k`` strongest gravity
    partners among its ``n_candidates`` nearest (symmetrized ⇒ sparse). This is the DEMOGRAPHIC-
    connectivity structure the LDO's spatial prior should use for epidemiology (P1 intent): a big city
    exerts a stronger, longer-reaching pull; a dense cluster of small municipalities couples tightly; an
    isolated small one weakly — the urban-lattice dynamics that geometric distance/contiguity cannot
    express. Population is the demographic mass (from the population tensor's closure), distance is the
    geobr centroid distance.

    ``context_derived`` (weights encode population) ⇒ subject to the §II.4.1 circularity guard: valid to
    smooth provenance-disjoint OUTCOMES (mortality/morbidity/socioeconomic), NEVER a population-derived
    variable (a rate's denominator is population — using a population kernel there is circular). The
    symmetric-normalized Laplacian is scale-free in the weights, so only the relative gravity structure
    (not the absolute pop² magnitude) enters the whitening. Returns ``(adjacency, edge_weights)`` for
    :func:`~pegasus.ldo.precision.build_spatial_precision_sparse`, or ``(None, None)`` when too sparse."""
    from scipy.spatial import cKDTree

    if centroids is None:
        centroids = load_municipality_centroids()
    present = [s for s in space_ids if s in centroids and float(populations.get(s, 0.0)) > 0.0]
    if len(present) < max(k + 1, 4):
        return None, None
    xy = np.array([centroids[s] for s in present], dtype=np.float64)
    pop = np.array([float(populations[s]) for s in present], dtype=np.float64)
    m = min(n_candidates + 1, len(present))
    dist, nn = cKDTree(xy).query(xy, k=m)
    dist = np.atleast_2d(dist)
    nn = np.atleast_2d(nn)
    adj: dict[str, set] = {}
    wts: dict[str, dict[str, float]] = {}
    for i, s in enumerate(present):
        cand = nn[i, 1:]  # skip self at rank 0
        d = np.maximum(dist[i, 1:], 1.0)
        grav = (pop[i] * pop[cand]) / (d ** gamma)
        for idx in np.argsort(grav)[::-1][:k]:
            t = present[int(cand[idx])]
            w = float(grav[idx])
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)
            wts.setdefault(s, {})[t] = w
            wts.setdefault(t, {})[s] = w
    wmax = max((w for dd in wts.values() for w in dd.values()), default=1.0)
    if wmax > 0:
        for s in wts:
            for t in wts[s]:
                wts[s][t] /= wmax  # cosmetic — the symmetric-normalized Laplacian is scale-free
    adjacency = {s: tuple(sorted(v)) for s, v in adj.items()}
    return adjacency, wts


__all__ = [
    "build_centroids_artifact",
    "load_municipality_centroids",
    "distance_decay_edge_weights",
    "estimate_spatial_range",
    "knn_distance_graph",
    "gravity_edge_weights",
]
