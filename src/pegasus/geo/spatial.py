"""Geographic spatial statistics over the real municipality adjacency (MSD-II §II.4).

Single source of truth for Moran's I and spatial effective sample size, computed on
the declared ``SpatialWeightGraph`` adjacency (queen contiguity / distance), not on a
1-D cell-ordering proxy. Consumers (Q-tensor, LDO) call these instead of re-deriving
chain contiguity.
"""

from __future__ import annotations

import numpy as np

from pegasus.geo.spatial_graph import (
    SpatialWeightGraph,
    assert_spatial_legality,
)


def _align(values, graph: SpatialWeightGraph) -> tuple[np.ndarray, np.ndarray]:
    """Return (x over graph.node_ids order, finite mask). ``values`` may be a mapping
    node_id -> value, or a positional sequence aligned to ``graph.node_ids``."""
    n = graph.n
    x = np.full(n, np.nan, dtype=np.float64)
    if hasattr(values, "get") and hasattr(values, "keys"):
        idx = graph.index
        for k, v in values.items():
            i = idx.get(str(k))
            if i is not None and v is not None:
                try:
                    x[i] = float(v)
                except (TypeError, ValueError):
                    continue
    else:
        arr = np.asarray(list(values), dtype=np.float64)
        m = min(arr.shape[0], n)
        x[:m] = arr[:m]
    return x, np.isfinite(x)


def _edge_arrays(graph: SpatialWeightGraph) -> tuple[np.ndarray, np.ndarray]:
    """Flat directed (src, dst) node-index arrays from the sparse adjacency — matrix-free
    so national S (thousands of nodes) never densifies to an S×S weight matrix."""
    idx = graph.index
    src: list[int] = []
    dst: list[int] = []
    for node, nbrs in graph._adjacency.items():
        i = idx.get(node)
        if i is None:
            continue
        for other in nbrs:
            j = idx.get(other)
            if j is not None and j != i:
                src.append(i)
                dst.append(j)
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def moran_i(values, graph: SpatialWeightGraph, *, variable_provenance=None) -> float | None:
    """Row-standardized-W Moran's I over the true adjacency, computed matrix-free.

    ``I = (n/S0)·(Σᵢⱼ wᵢⱼ zᵢ zⱼ)/(Σᵢ zᵢ²)`` with full-degree row-standardized ``W`` restricted
    to the observed sub-support. None when <3 observed cells, zero variance, or no edges.
    """
    assert_spatial_legality(graph, variable_provenance)
    x, mask = _align(values, graph)
    n_obs = int(mask.sum())
    if n_obs < 3:
        return None
    src, dst = _edge_arrays(graph)
    if src.size == 0:
        return None
    deg = np.bincount(src, minlength=graph.n).astype(np.float64)  # full-graph out-degree
    z = np.where(mask, x - x[mask].mean(), 0.0)
    denom = float(z[mask] @ z[mask])
    keep = mask[src] & mask[dst]
    if denom <= 0 or not keep.any():
        return None
    s, d = src[keep], dst[keep]
    w = 1.0 / deg[s]
    w_total = float(w.sum())
    if w_total <= 0:
        return None
    cross = float((w * z[s] * z[d]).sum())
    return float((n_obs / w_total) * (cross / denom))


def effective_n(values, graph: SpatialWeightGraph, *, variable_provenance=None) -> float | None:
    """Moran/Griffith spatial effective sample size.

    Maps Moran's I to a first-order autoregressive ``rho`` (clipped to a stable open
    interval) and returns ``n·(1-rho)/(1+rho)``, floored at 1 and capped at n — positive
    spatial autocorrelation shrinks the count of independent observations. ``rho<=0`` (no
    positive autocorrelation) leaves n_eff at n.
    """
    x, mask = _align(values, graph)
    n = int(mask.sum())
    if n < 3:
        return float(n) if n > 0 else None
    I = moran_i(values, graph, variable_provenance=variable_provenance)
    if I is None:
        return float(n)
    rho = float(np.clip(I, 0.0, 0.99))
    return float(min(float(n), max(1.0, n * (1.0 - rho) / (1.0 + rho))))


def weight_view(graph: SpatialWeightGraph, kind: str = "row_standardized", *, variable_provenance=None) -> np.ndarray:
    """Legality-checked weight matrix over ``graph.node_ids`` (contiguity/distance/…)."""
    assert_spatial_legality(graph, variable_provenance)
    return graph.view(kind)


__all__ = ["moran_i", "effective_n", "weight_view"]
