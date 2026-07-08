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


__all__ = ["build_centroids_artifact", "load_municipality_centroids", "distance_decay_edge_weights"]
