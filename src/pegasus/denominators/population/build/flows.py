"""Migration flow-field reconstruction + affinity kernel persistence (MSD §2.8.7 flow layer)."""

from __future__ import annotations

import warnings
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
) -> tuple[str | None, str | None, tuple[dict[str, Any], ...], str | None]:
    """Reconstruct O→D migration flows + affinity kernel from the tensor's own net
    residual and closure populations (MSD §2.8.7 flow layer). Returns
    ``(flows_path, affinity_path, per_year_manifests, skip_reason)``.

    ``skip_reason`` is ``None`` on success and on the ONE legitimately-empty case (no
    net-migration signal at all). It is a non-None reason string — and a ``RuntimeWarning`` is
    emitted — whenever a reconstruction was *attempted but could not run*, so the absence of the
    migration-affinity field is never silent (§V "never silently cap/degrade"). The dominant such
    case is national scale: the queen-contiguity candidate-pair count (~16k at 5570 munis) exceeds
    ``MAX_DENSE_FLOW_PAIRS`` and the dense reconstruction refuses — downstream then falls back to
    plain contiguity, and the caller records this reason in the build manifest."""
    if migration_locality_totals is None:
        return None, None, (), None  # no net-migration signal → nothing to reconstruct (legitimately quiet)
    from pegasus.geo.migration_affinity import build_migration_affinity_graph
    from pegasus.geo.spatial_graph import structural_cod6_adjacency
    from pegasus.denominators.population.migration import MigrationFlowError, reconstruct_migration_flows

    def _skip(reason: str) -> tuple[None, None, tuple[()], str]:
        warnings.warn(
            f"migration flow reconstruction skipped ({reason}); the migration-affinity spatial "
            f"kernel is absent for this build and downstream falls back to structural contiguity.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, (), reason

    try:
        full_adjacency = structural_cod6_adjacency(contiguity_graph_id)
    except Exception as exc:  # noqa: BLE001 -- surface, don't swallow
        return _skip(f"contiguity_graph_unavailable:{contiguity_graph_id}:{type(exc).__name__}")
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
    except MigrationFlowError as exc:
        # The dominant national case: candidate-pair count > MAX_DENSE_FLOW_PAIRS → the dense
        # backend refuses. Surface it (§V) instead of returning a silent no-op that reads
        # downstream as "no migration signal". Re-enabling national flows needs a sparse backend.
        return _skip(f"dense_reconstruction_refused:{exc}")
    if not reconstructions:
        return _skip("no_flows_reconstructed")

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
    return str(flows_path), str(affinity_path), tuple(rec.as_manifest() for rec in reconstructions), None


__all__ = [
    "_reconstruct_and_persist_migration_flows",
]
