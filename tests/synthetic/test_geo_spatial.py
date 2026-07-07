"""Geographic Moran's I / effective-n over the true adjacency (Theme-6 MORAN-ORDERING-PROXY-01).

Proves the consolidated geo.spatial API measures autocorrelation on the declared
SpatialWeightGraph adjacency — not on cell ordering: a field smooth over the graph
scores high, the same values shuffled across nodes score ~0, and effective_n shrinks
below n only under positive autocorrelation.
"""

from __future__ import annotations

import numpy as np

from pegasus.geo.spatial import effective_n, moran_i
from pegasus.geo.spatial_graph import SpatialWeightGraph


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
        node_ids=tuple(nodes),
        _adjacency={k: tuple(v) for k, v in adj.items()},
    )


def test_moran_geographic_vs_shuffled_and_effective_n():
    side = 8
    graph = _grid_graph(side)
    # smooth field over the grid: value = row + col (varies gently between neighbours)
    smooth = {f"{r}_{c}": float(r + c) for r in range(side) for c in range(side)}

    I_smooth = moran_i(smooth, graph)
    assert I_smooth is not None and I_smooth > 0.5, f"smooth grid field should be strongly autocorrelated, got {I_smooth}"

    # same values, shuffled across nodes: destroys geographic structure → Moran ~ 0
    rng = np.random.default_rng(0)
    vals = list(smooth.values())
    rng.shuffle(vals)
    shuffled = {n: v for n, v in zip(graph.node_ids, vals)}
    I_shuffled = moran_i(shuffled, graph)
    assert I_shuffled is not None and abs(I_shuffled) < 0.25, f"shuffled field should be ~uncorrelated, got {I_shuffled}"
    assert I_smooth > I_shuffled + 0.3

    n = graph.n
    neff_smooth = effective_n(smooth, graph)
    neff_shuffled = effective_n(shuffled, graph)
    assert neff_smooth < n, f"positive autocorrelation must shrink n_eff below n={n}, got {neff_smooth}"
    assert neff_shuffled >= neff_smooth
    assert neff_shuffled <= n + 1e-9

    # independent noise → effective_n ~ n (no shrinkage floor artifacts)
    indep = {n_id: float(rng.standard_normal()) for n_id in graph.node_ids}
    neff_indep = effective_n(indep, graph)
    assert neff_indep > 0.8 * n, f"independent field n_eff should be near n, got {neff_indep} vs n={n}"
