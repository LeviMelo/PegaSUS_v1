"""Migration-flow spatial affinity kernel (MSD-II §II.4 context_derived graph).

Turns reconstructed origin→destination migration flows into a functional adjacency
graph between municipalities: two cities are "close" to the degree that people move
between them, regardless of whether they share a border. This is a deeper notion of
adjacency than geographic contiguity for demographic/health diffusion — a migration
corridor (e.g. capital ↔ satellite town) can be far stronger than a shared border
with an empty neighbour.

The kernel:

    affinity(i, j) = [ F(i→j) + F(j→i) ]  /  sqrt(Pop_i · Pop_j)

Symmetrized (migration connectivity is undirected) and mass-normalized so it is a
*propensity* — the per-capita tendency to interchange population — rather than raw
volume, which would just rank the biggest cities first. Flows are summed across the
run's years for a stable kernel.

By construction this graph's weights depend on the ``migration_flow`` and
``population`` substantive variables, so it is ``legality_class="context_derived"``
with that provenance and is subject to the §II.4.1 circularity guard
(``geo.spatial_graph.assert_spatial_legality``): it may not spatially-smooth any
variable that itself derives from migration or population (e.g. the population
denominator tensor), only provenance-disjoint outcomes.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

from pegasus.geo.spatial_graph import SpatialWeightGraph
from pegasus.sidra.population_cube.migration import MigrationFlowReconstruction


MIGRATION_AFFINITY_PROVENANCE = ("migration_flow", "population")


def build_migration_affinity_graph(
    reconstructions: Iterable[MigrationFlowReconstruction],
    populations_by_node: dict[str, float],
    *,
    graph_id: str = "migration_affinity",
    min_affinity: float = 0.0,
    top_k: int | None = None,
) -> SpatialWeightGraph:
    """Build a ``context_derived`` migration-affinity SpatialWeightGraph.

    ``populations_by_node`` is a representative (e.g. mid-window or mean) population
    per municipality used for the mass normalization. ``min_affinity`` prunes weak
    edges; ``top_k`` (if set) keeps only each node's strongest ``k`` edges — both
    keep the graph sparse for the downstream GMRF/Laplacian.
    """
    # Sum directed flows across all reconstructed years, then symmetrize.
    directed: dict[tuple[str, str], float] = {}
    nodes: set[str] = set()
    for rec in reconstructions:
        nodes.update(rec.node_ids)
        for (i, j), value in rec.flows.items():
            directed[(i, j)] = directed.get((i, j), 0.0) + float(value)
    nodes.update(populations_by_node)

    undirected: dict[tuple[str, str], float] = {}
    for (i, j), value in directed.items():
        key = (i, j) if i <= j else (j, i)
        undirected[key] = undirected.get(key, 0.0) + value

    weights: dict[str, dict[str, float]] = {}
    for (i, j), flow in undirected.items():
        pop_i = max(float(populations_by_node.get(i, 0.0)), 0.0)
        pop_j = max(float(populations_by_node.get(j, 0.0)), 0.0)
        denom = sqrt(pop_i * pop_j)
        affinity = (flow / denom) if denom > 0 else 0.0
        if affinity <= min_affinity:
            continue
        weights.setdefault(i, {})[j] = affinity
        weights.setdefault(j, {})[i] = affinity

    if top_k is not None:
        pruned: dict[str, dict[str, float]] = {}
        for node, edges in weights.items():
            kept = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            pruned[node] = dict(kept)
        # Re-symmetrize after per-node top-k (keep an edge if either endpoint kept it).
        weights = {}
        for node, edges in pruned.items():
            for other, w in edges.items():
                weights.setdefault(node, {})[other] = w
                weights.setdefault(other, {})[node] = w

    adjacency = {node: tuple(sorted(edges)) for node, edges in weights.items()}
    node_ids = tuple(sorted(nodes))
    for node in node_ids:
        adjacency.setdefault(node, ())
    return SpatialWeightGraph(
        graph_id=graph_id,
        legality_class="context_derived",
        provenance=MIGRATION_AFFINITY_PROVENANCE,
        node_ids=node_ids,
        _adjacency=adjacency,
        directed=False,
        _weights=weights or None,
    )


__all__ = ["MIGRATION_AFFINITY_PROVENANCE", "build_migration_affinity_graph"]
