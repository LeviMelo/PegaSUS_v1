"""SpatialWeightGraph — one declared base graph, many views (MSD-II §II.4).

MII-SPG-01/02/03. A single structural spatial graph (municipality queen
contiguity by default) is declared in ``config/registries/spatial/spatial_graphs.yaml``
and loaded here. Every spatial consumer draws its structure from this one object
rather than re-deriving ad-hoc adjacency:

- Moran's I               → ``view("row_standardized")``
- ICAR / CAR precision    → ``view("symmetric")`` / ``view("laplacian")``
- CTR / ST-DFM / LDO prior→ ``view("laplacian")`` (GMRF precision ``κI + L``)
- HSIC spatial null       → ``blocks(n)``
- cross-fit folds         → ``blocks(n)``

Circularity guard (§II.4.1): a ``context_derived`` graph (weights depend on a
substantive variable) MUST be rejected for any test/edge whose variable shares
the graph's provenance. ``assert_spatial_legality`` enforces this; the default
``structural`` contiguity graph is provenance-disjoint from all variables and so
is always legal.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from pegasus.geo.adjacency import load_adjacency


class SpatialGraphError(ValueError):
    """Raised when a spatial graph is missing, malformed, or illegally applied."""


class SpatialCircularityError(SpatialGraphError):
    """Raised when a context-derived graph shares provenance with the tested variable (§II.4.1)."""


_VIEWS = ("binary", "symmetric", "row_standardized", "laplacian")


@dataclass(frozen=True)
class SpatialWeightGraph:
    graph_id: str
    legality_class: str
    provenance: tuple[str, ...]
    node_ids: tuple[str, ...]
    _adjacency: dict[str, tuple[str, ...]]
    directed: bool = False

    @property
    def n(self) -> int:
        return len(self.node_ids)

    @property
    def index(self) -> dict[str, int]:
        return {node: i for i, node in enumerate(self.node_ids)}

    def neighbors(self, node: str) -> tuple[str, ...]:
        return self._adjacency.get(node, ())

    def view(self, kind: str = "binary") -> np.ndarray:
        """Return the requested weight-matrix view over ``node_ids`` order."""
        if kind not in _VIEWS:
            raise SpatialGraphError(f"unknown spatial view '{kind}'; expected one of {_VIEWS}")
        idx = self.index
        n = self.n
        binary = np.zeros((n, n), dtype=np.float64)
        for node, neighbours in self._adjacency.items():
            i = idx.get(node)
            if i is None:
                continue
            for other in neighbours:
                j = idx.get(other)
                if j is not None and i != j:
                    binary[i, j] = 1.0
        if kind in {"binary", "symmetric"}:
            # Undirected contiguity is already symmetric; enforce it defensively.
            return np.maximum(binary, binary.T) if not self.directed else binary
        if kind == "row_standardized":
            degree = binary.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                row = np.where(degree > 0, binary / degree, 0.0)
            return row
        # laplacian: L = D - A (combinatorial), the GMRF precision skeleton κI + L.
        adjacency = np.maximum(binary, binary.T)
        degree = np.diag(adjacency.sum(axis=1))
        return degree - adjacency

    def blocks(self, n_blocks: int) -> list[list[str]]:
        """Partition nodes into ``n_blocks`` spatially-contiguous groups.

        Used for HSIC spatial nulls and cross-fit folds: a BFS-grown partition
        keeps each block geographically compact (neighbours stay together), which
        preserves spatial dependence structure within a block for block-permutation
        and block-holdout schemes.
        """
        if n_blocks < 1:
            raise SpatialGraphError("n_blocks must be >= 1")
        if n_blocks == 1 or self.n == 0:
            return [list(self.node_ids)]
        # Ceil target so ~n_blocks balanced blocks result. Each block is filled to
        # ``target`` by seeding successive BFS trees from the remaining nodes, so
        # small disconnected components (islands) merge into a block rather than
        # each consuming a whole block slot (which would leave one giant block).
        target = -(-self.n // n_blocks)
        remaining = list(self.node_ids)
        remaining_set = set(remaining)
        blocks: list[list[str]] = []
        while remaining_set:
            block: list[str] = []
            while len(block) < target and remaining_set:
                seed = next(node for node in remaining if node in remaining_set)
                frontier = [seed]
                while frontier and len(block) < target:
                    node = frontier.pop(0)
                    if node not in remaining_set:
                        continue
                    remaining_set.discard(node)
                    block.append(node)
                    frontier.extend(nb for nb in self._adjacency.get(node, ()) if nb in remaining_set)
            blocks.append(block)
        return blocks


def _load_registry(root: str | Path) -> dict:
    path = Path(root) / "spatial/spatial_graphs.yaml"
    if not path.exists():
        raise SpatialGraphError(f"spatial/spatial_graphs.yaml missing under {root}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=16)
def load_spatial_graph(graph_id: str = "contiguity_queen", root: str = "config/registries") -> SpatialWeightGraph:
    """Load a declared spatial base graph and its adjacency artifact."""
    registry = _load_registry(root)
    spec = (registry.get("graphs") or {}).get(graph_id)
    if spec is None:
        raise SpatialGraphError(f"spatial graph '{graph_id}' not declared in spatial/spatial_graphs.yaml")
    artifact = Path(root) / str(spec["artifact"])
    adjacency = load_adjacency(artifact, require_symmetric=not bool(spec.get("directed", False)))
    nodes = set(adjacency)
    for neighbours in adjacency.values():
        nodes.update(neighbours)
    node_ids = tuple(sorted(nodes))
    return SpatialWeightGraph(
        graph_id=graph_id,
        legality_class=str(spec.get("legality_class", "structural")),
        provenance=tuple(str(p) for p in (spec.get("provenance") or ())),
        node_ids=node_ids,
        _adjacency={k: tuple(v) for k, v in adjacency.items()},
        directed=bool(spec.get("directed", False)),
    )


@lru_cache(maxsize=4)
def _structural_cod6_items(graph_id: str, root: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    graph = load_spatial_graph(graph_id, root)
    cod6: dict[str, set[str]] = {}
    for node in graph.node_ids:
        left = node[:6]
        for neighbour in graph.neighbors(node):
            right = neighbour[:6]
            if left != right:
                cod6.setdefault(left, set()).add(right)
    return tuple((key, tuple(sorted(values))) for key, values in sorted(cod6.items()))


def structural_graph_available(graph_id: str = "contiguity_queen", root: str = "config/registries") -> bool:
    """True when the structural spatial graph and its artifact are loadable.

    Lets spatial-mode selection know ICAR/GMRF is executable by default (the graph
    is a committed artifact) even when an intent declares no adjacency path.
    """
    try:
        return load_spatial_graph(graph_id, root).n > 0
    except Exception:
        return False


def structural_cod6_adjacency(
    graph_id: str = "contiguity_queen", root: str = "config/registries"
) -> dict[str, tuple[str, ...]]:
    """Structural contiguity as a ``municipality_cod6``-keyed adjacency dict.

    The SpatialWeightGraph is keyed by IBGE cod7; the EFG/PIRS panels key geography
    by ``municipality_cod6`` (cod7 minus the check digit). This is the single
    cod7→cod6 down-map used by every consumer that falls back to the structural
    default graph (Moran's I, ICAR, ...). Returns a fresh dict each call (callers
    must not mutate the shared cache).
    """
    return dict(_structural_cod6_items(graph_id, root))


def assert_spatial_legality(graph: SpatialWeightGraph, variable_provenance) -> None:
    """§II.4.1 circularity guard.

    A ``structural`` graph is always legal. A ``context_derived`` graph is illegal
    for any variable whose provenance overlaps the graph's provenance (the weights
    would encode the very quantity under test).
    """
    if graph.legality_class == "structural":
        return
    shared = set(graph.provenance) & set(str(p) for p in (variable_provenance or ()))
    if shared:
        raise SpatialCircularityError(
            "context_derived spatial weight shares provenance with the tested variable: "
            f"{sorted(shared)}"
        )


__all__ = [
    "SpatialWeightGraph",
    "SpatialGraphError",
    "SpatialCircularityError",
    "load_spatial_graph",
    "structural_cod6_adjacency",
    "structural_graph_available",
    "assert_spatial_legality",
]
