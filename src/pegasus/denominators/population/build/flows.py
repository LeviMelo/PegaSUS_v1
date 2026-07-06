"""Migration flow-field reconstruction + affinity kernel persistence (MSD §2.8.7 flow layer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def _reconstruct_and_persist_migration_flows(
    *,
    localities: tuple[str, ...],
    periods: tuple[str, ...],
    closure: list[float | None],
    migration_locality_totals: tuple[float | None, ...] | None,
    shape: tuple[int, int, int, int, int],
    out_path: Path,
    contiguity_graph_id: str,
    max_hops: int,
) -> tuple[str | None, str | None, tuple[dict[str, Any], ...]]:
    """Reconstruct O→D migration flows + affinity kernel from the tensor's own net
    residual and closure populations (MSD §2.8.7 flow layer). Returns
    ``(flows_path, affinity_path, per_year_manifests)``; a no-op ``(None, None, ())``
    when there is no net-migration signal or the contiguity graph is unavailable."""
    if migration_locality_totals is None:
        return None, None, ()
    from pegasus.geo.migration_affinity import build_migration_affinity_graph
    from pegasus.geo.spatial_graph import structural_cod6_adjacency
    from pegasus.denominators.population.migration import MigrationFlowError, reconstruct_migration_flows

    try:
        full_adjacency = structural_cod6_adjacency(contiguity_graph_id)
    except Exception:
        return None, None, ()
    locality_set = set(localities)
    adjacency = {loc: tuple(n for n in full_adjacency.get(loc, ()) if n in locality_set) for loc in localities}

    t_count = shape[1]
    populations_by_year: dict[str, dict[str, float]] = {}
    net_by_year: dict[str, dict[str, float]] = {}
    for t, period in enumerate(periods):
        pops: dict[str, float] = {}
        nets: dict[str, float] = {}
        for s, loc in enumerate(localities):
            c = closure[s * t_count + t]
            if c is not None:
                pops[loc] = float(c)
            m = migration_locality_totals[s * t_count + t]
            if m is not None:
                nets[loc] = float(m)
        populations_by_year[period] = pops
        net_by_year[period] = nets

    try:
        reconstructions = reconstruct_migration_flows(
            nodes=list(localities),
            populations_by_year=populations_by_year,
            net_by_year=net_by_year,
            adjacency=adjacency,
            max_hops=max_hops,
        )
    except MigrationFlowError:
        return None, None, ()
    if not reconstructions:
        return None, None, ()

    flow_rows = [
        {"year": int(rec.year), "origin_cod6": i, "destination_cod6": j, "flow": value}
        for rec in reconstructions for (i, j), value in rec.flows.items()
    ]
    flows_path = out_path.with_suffix(".migration_flows.parquet")
    pl.DataFrame(flow_rows, schema={"year": pl.Int64, "origin_cod6": pl.Utf8, "destination_cod6": pl.Utf8, "flow": pl.Float64}).write_parquet(flows_path)

    # Representative population per node (mean over years present) for the mass
    # normalization of the affinity kernel.
    pop_accum: dict[str, list[float]] = {}
    for pops in populations_by_year.values():
        for loc, value in pops.items():
            pop_accum.setdefault(loc, []).append(value)
    populations_by_node = {loc: (sum(v) / len(v)) for loc, v in pop_accum.items() if v}
    graph = build_migration_affinity_graph(reconstructions, populations_by_node)
    edge_rows: list[dict[str, Any]] = []
    weight_matrix = graph.view("weight")
    index = graph.index
    for i in graph.node_ids:
        for j in graph.neighbors(i):
            if i < j:  # undirected: emit each edge once
                edge_rows.append({"source_cod6": i, "target_cod6": j, "affinity": float(weight_matrix[index[i], index[j]])})
    affinity_path = out_path.with_suffix(".migration_affinity.parquet")
    pl.DataFrame(edge_rows, schema={"source_cod6": pl.Utf8, "target_cod6": pl.Utf8, "affinity": pl.Float64}).write_parquet(affinity_path)
    return str(flows_path), str(affinity_path), tuple(rec.as_manifest() for rec in reconstructions)


__all__ = [
    "_reconstruct_and_persist_migration_flows",
]
