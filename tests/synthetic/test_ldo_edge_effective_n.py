"""Per-edge effective-n Fisher-z SE (A13, W1-pt2).

The Fisher-z SE denominator 1/√(n_eff−3) must be sized PER EDGE: spatial autocorrelation
over the true adjacency and reliability down-weighting both shrink an edge's effective-n
(inflate its uncertainty). It is no longer one raw observed-cell count shared by every edge.
"""

from __future__ import annotations

import numpy as np

from pegasus.geo.spatial_graph import SpatialWeightGraph
from pegasus.ldo.edges import _edge_n_eff, _spatial_deflation_map, to_link_records
from pegasus.ldo.lags import LaggedFit
from pegasus.ldo.lowrank import SparseLowRankFit
from pegasus.ldo.margins import GaussianField


def _grid_graph(side: int) -> SpatialWeightGraph:
    nodes = [f"{r}_{c}" for r in range(side) for c in range(side)]
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    for r in range(side):
        for c in range(side):
            here = f"{r}_{c}"
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    adj[here].append(f"{rr}_{cc}")
    return SpatialWeightGraph(
        graph_id="grid", legality_class="structural", provenance=(),
        node_ids=tuple(nodes), _adjacency={k: tuple(v) for k, v in adj.items()},
    )


def _field(Z: np.ndarray, node_ids, W=None) -> GaussianField:
    p, S, T = Z.shape
    return GaussianField(
        variables=tuple("ABCD"[:p]), space_ids=tuple(node_ids), time_ids=tuple(range(T)),
        Z=Z, W=np.ones_like(Z) if W is None else W, resolution="year",
    )


def test_spatial_autocorrelation_shrinks_edge_n_eff():
    side = 8
    graph = _grid_graph(side)
    S, T = side * side, 6
    rng = np.random.default_rng(0)

    # Endpoints A,B smooth over the grid (strong positive autocorrelation).
    base = np.array([[float(r + c)] for r in range(side) for c in range(side)])  # (S,1)
    smooth = base + 0.05 * rng.standard_normal((S, T))
    Z_ac = np.stack([smooth, smooth + 0.3 * rng.standard_normal((S, T))])
    # Same coverage/reliability but spatially independent endpoints.
    Z_iid = rng.standard_normal((2, S, T))

    field_ac = _field(Z_ac, graph.node_ids)
    field_iid = _field(Z_iid, graph.node_ids)

    n_ac = _edge_n_eff(field_ac, "A", "B", _spatial_deflation_map(field_ac, graph, None))
    n_iid = _edge_n_eff(field_iid, "A", "B", _spatial_deflation_map(field_iid, graph, None))

    assert n_ac < n_iid, f"autocorrelated edge must earn smaller effective-n: {n_ac} vs {n_iid}"
    # Larger SE for the autocorrelated edge (1/√(n−3) decreasing in n).
    se_ac = 1.0 / np.sqrt(max(n_ac - 3, 1))
    se_iid = 1.0 / np.sqrt(max(n_iid - 3, 1))
    assert se_ac > se_iid


def test_reliability_downweight_shrinks_edge_n_eff_without_graph():
    S, T = 30, 6
    node_ids = tuple(f"{27}{i:05d}"[:7] for i in range(S))
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((2, S, T))
    field_reliable = _field(Z, node_ids)                 # W = 1 everywhere
    W_down = np.full((2, S, T), 0.05)                    # heavily reconstructed cells
    field_down = _field(Z, node_ids, W=W_down)

    n_reliable = _edge_n_eff(field_reliable, "A", "B", None)
    n_down = _edge_n_eff(field_down, "A", "B", None)

    assert n_down < n_reliable
    # No-graph fallback is the reliability-weighted joint cell count.
    assert n_reliable == float(S * T)


def _lagged_fit(variables, links, contemporaneous):
    p = len(variables)
    fit = SparseLowRankFit(
        S=np.eye(p), L=np.zeros((p, p)), precision=np.eye(p),
        factor_loadings=np.zeros((p, 0)), factor_values=np.zeros(0),
        converged=True, iterations=1, numerical_error=0.0,
    )
    return LaggedFit(
        variables=tuple(variables), K=1, fit=fit,
        lagged_links=list(links), contemporaneous=list(contemporaneous),
    )


def test_records_carry_distinct_per_edge_n_eff():
    # Two edges with DIFFERENT endpoint reliability → n_eff must differ across records
    # (it is no longer a single global constant).
    S, T = 24, 6
    node_ids = tuple(f"{27}{i:05d}"[:7] for i in range(S))
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((3, S, T))          # A, B, C
    W = np.ones((3, S, T))
    W[2] = 0.02                                 # C rests on heavily down-weighted cells
    field = _field(Z, node_ids, W=W)

    fit = _lagged_fit(("A", "B", "C"), links=[], contemporaneous=[("A", "B", 0.5), ("A", "C", 0.5)])
    recs = to_link_records(fit, field=field)
    by_pair = {(r.source_var, r.target_var): r for r in recs}

    ab = by_pair[("A", "B")]
    ac = by_pair[("A", "C")]
    assert ab.n_eff is not None and ac.n_eff is not None
    assert ac.n_eff < ab.n_eff, "the down-weighted-endpoint edge must carry a smaller n_eff"
    assert ac.uncertainty > ab.uncertainty, "smaller n_eff must inflate the Fisher-z SE"
    assert ab.n_eff != ac.n_eff  # not one global constant


def test_per_edge_off_is_uniform_noop():
    # per_edge_n_eff=False → uncertainty is the single global-n value for every edge (prior behaviour).
    S, T = 24, 6
    node_ids = tuple(f"{27}{i:05d}"[:7] for i in range(S))
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((3, S, T))
    W = np.ones((3, S, T)); W[2] = 0.02
    field = _field(Z, node_ids, W=W)

    fit = _lagged_fit(("A", "B", "C"), links=[], contemporaneous=[("A", "B", 0.5), ("A", "C", 0.5)])
    recs = to_link_records(fit, field=field, per_edge_n_eff=False)
    uncs = {r.uncertainty for r in recs}
    assert len(uncs) == 1, "with per-edge off, every edge shares the global uncertainty"

    import math
    n_global = int(np.isfinite(field.Z).any(axis=0).sum())
    expected = 1.0 / math.sqrt(max(n_global - 3, 1))
    assert abs(next(iter(uncs)) - expected) < 1e-12
