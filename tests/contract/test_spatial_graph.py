"""SPG-01 — SpatialWeightGraph contract (MSD-III §II.6 guard, I.4).

Pins the one declared spatial base graph and its derived views (MII-SPG-01, already
built in ``geo/spatial_graph.py``): the binary/row-standardized/symmetric/laplacian
views, the spatially-contiguous ``blocks(n)`` partition consumed by HSIC nulls and
cross-fit folds, and the §II.4.1 circularity guard that refuses a context-derived
graph as a same-provenance prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from pegasus.geo.spatial_graph import (
    SpatialCircularityError,
    SpatialWeightGraph,
    assert_spatial_legality,
    load_spatial_graph,
)

# A 5-node path graph a-b-c-d-e (undirected, unweighted contiguity).
_ADJ = {
    "a": ("b",),
    "b": ("a", "c"),
    "c": ("b", "d"),
    "d": ("c", "e"),
    "e": ("d",),
}


def _fixture_graph(legality_class: str = "structural", provenance: tuple[str, ...] = ()) -> SpatialWeightGraph:
    return SpatialWeightGraph(
        graph_id="fixture_path5",
        legality_class=legality_class,
        provenance=provenance,
        node_ids=("a", "b", "c", "d", "e"),
        _adjacency=_ADJ,
    )


def test_views_on_five_node_fixture() -> None:
    g = _fixture_graph()

    binary = g.view("binary")
    assert binary.shape == (5, 5)
    assert np.allclose(binary, binary.T)          # undirected → symmetric
    assert np.all(np.diag(binary) == 0.0)         # no self-loops
    assert binary[0, 1] == 1.0 and binary[0, 2] == 0.0

    row = g.view("row_standardized")
    # every non-isolated row sums to 1; b (degree 2) splits its weight evenly
    assert row.sum(axis=1) == pytest.approx([1, 1, 1, 1, 1])
    assert row[1, 0] == pytest.approx(0.5)

    lap = g.view("laplacian")                      # L = D - W
    assert np.allclose(lap, lap.T)
    assert np.allclose(lap.sum(axis=1), 0.0)       # GMRF Laplacian rows sum to zero
    assert lap[1, 1] == pytest.approx(2.0)         # degree of b
    assert lap[1, 0] == pytest.approx(-1.0)


def test_blocks_partition_all_nodes() -> None:
    g = _fixture_graph()
    blocks = g.blocks(2)
    flat = [node for block in blocks for node in block]
    assert sorted(flat) == list(g.node_ids)        # covers every node
    assert len(flat) == len(set(flat))             # disjoint
    assert 1 < len(blocks) <= 3                     # roughly the requested count


def test_circularity_guard() -> None:
    # structural graphs are always legal, regardless of the tested variable
    assert_spatial_legality(_fixture_graph(legality_class="structural"), ("migration_flow", "gdp"))

    # a context-derived graph sharing provenance with the tested variable is refused
    ctx = _fixture_graph(legality_class="context_derived", provenance=("migration_flow",))
    with pytest.raises(SpatialCircularityError):
        assert_spatial_legality(ctx, ("migration_flow", "deaths"))

    # ...but is legal for a provenance-disjoint variable
    assert_spatial_legality(ctx, ("gdp", "sanitation"))


def test_real_contiguity_graph_is_structural_and_legal() -> None:
    g = load_spatial_graph("contiguity_queen")
    assert g.legality_class == "structural"
    assert g.n > 5000                              # ~5570 Brazilian municipalities
    # structural default is legal as a prior for any substantive variable
    assert_spatial_legality(g, ("population", "deaths", "gdp"))
    # a real node has real neighbors (queen contiguity), never an ordering proxy
    some_node = next(node for node in g.node_ids if g.neighbors(node))
    assert len(g.neighbors(some_node)) >= 1
