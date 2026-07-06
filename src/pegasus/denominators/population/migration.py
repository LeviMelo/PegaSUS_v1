"""Latent origin→destination migration-flow reconstruction (MSD §2.8.7 flow layer).

The net-migration residual (``build._migration_residual_totals``) gives, per
municipality-year, only the *net* balance ``inflow - outflow`` — a single number
per node. This module reconstructs the far richer directed **flow field**
``F(i→j, t)`` (who moves from where to where) from those net marginals, using:

  1. a **gravity structural prior** — the production-constrained spatial-interaction
     model: each origin ``i`` emits ``base_rate · Pop_i`` migrants, allocated over
     destinations ``j`` by ``Pop_j / hops(i,j)^γ`` (mass-attracting, distance-decaying);
  2. the observed **net marginal** ``S F = net`` (the balancing identity); and
  3. an optional **census O→D anchor** (the genuine bilateral flows IBGE publishes
     decennially) pinning the structure at census years.

Reconstruction is the generic ``migration_flow_instance`` CTR (all-quadratic, convex,
non-negative). Candidate flows are restricted to municipality pairs within
``max_hops`` of the contiguity graph — both a computational necessity (dense O→D is
N² unknowns) and a demographic truth (migration is overwhelmingly short-range).

Honesty: without a census anchor this is a gravity-structured, net-consistent
*estimate* (the net marginal fixes each node's level; gravity fixes the relative
allocation among neighbours), not an observation — flagged accordingly. With a
census anchor it is an interpolation pinned to real bilateral flows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pegasus.denominators.reconstruction.instances import migration_flow_instance, net_flow_operator
from pegasus.denominators.reconstruction.problem import solve_ctr


# Guard: the generic CTR builds a dense (P×P) Hessian for its exact solve. Above
# this many candidate directed pairs the reconstruction refuses rather than risk
# OOM; a sparse backend (mirroring sparse_admm) is the documented scaling path.
MAX_DENSE_FLOW_PAIRS = 8000


class MigrationFlowError(ValueError):
    """Raised when a migration-flow reconstruction cannot be posed legally."""


@dataclass(frozen=True)
class MigrationFlowReconstruction:
    """One year's reconstructed directed flow field over candidate pairs."""

    year: str
    node_ids: tuple[str, ...]
    flows: dict[tuple[str, str], float]  # (origin_cod6, destination_cod6) -> persons
    converged: bool
    net_residual_l1: float               # ||S F_hat - net_observed||_1 (balance mismatch)
    gross_flow: float                    # total persons moved (sum of F)
    anchored: bool                       # whether a census O→D anchor pinned this year
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "n_nodes": len(self.node_ids),
            "n_pairs": len(self.flows),
            "converged": self.converged,
            "net_residual_l1": self.net_residual_l1,
            "gross_flow": self.gross_flow,
            "anchored": self.anchored,
            "warnings": list(self.warnings),
        }


def hop_distances(
    adjacency: dict[str, tuple[str, ...]], nodes: list[str], *, max_hops: int | None = None
) -> dict[tuple[str, str], int]:
    """Shortest hop count over the (undirected) contiguity adjacency.

    BFS from each node. Unreachable pairs are omitted (no key). Distance to self
    is 0. This is the distance proxy for the gravity prior — no coordinates exist,
    and contiguity-hop distance is a sound monotone stand-in for real distance.

    ``max_hops`` bounds the BFS radius: only pairs within ``max_hops`` are recorded.
    The gravity prior and candidate-pair support use distances in ``1..max_hops`` only,
    so bounding loses nothing while turning the memory from O(N²) (a national all-pairs
    dict is ~5,570² ≈ 31M entries, multi-GB) into O(N · neighbours-within-max_hops).
    Left unbounded (``None``) it is the full all-pairs distance (small graphs / tests).
    """
    node_set = set(nodes)
    out: dict[tuple[str, str], int] = {}
    for source in nodes:
        seen = {source: 0}
        queue: deque[str] = deque([source])
        while queue:
            node = queue.popleft()
            d = seen[node]
            if max_hops is not None and d >= max_hops:
                continue  # do not expand past the radius the gravity prior can use
            for neighbour in adjacency.get(node, ()):  # type: ignore[union-attr]
                if neighbour in node_set and neighbour not in seen:
                    seen[neighbour] = d + 1
                    queue.append(neighbour)
        for target, dist in seen.items():
            out[(source, target)] = dist
    return out


def _candidate_pairs(nodes: list[str], hops: dict[tuple[str, str], int], max_hops: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            d = hops.get((i, j))
            if d is not None and 1 <= d <= max_hops:
                pairs.append((i, j))
    return pairs


def _gravity_prior(
    pairs: list[tuple[str, str]],
    populations: dict[str, float],
    hops: dict[tuple[str, str], int],
    *,
    base_rate: float,
    distance_decay: float,
) -> np.ndarray:
    """Production-constrained gravity expectation per directed pair.

    Origin ``i`` emits ``base_rate · Pop_i`` migrants split over its candidate
    destinations by ``Pop_j / hops(i,j)^decay`` (normalized per origin). So the
    prior's per-origin outflow mass is demographically meaningful (a fraction of
    the origin population), giving the whole flow field a real absolute scale that
    the net marginal then corrects.
    """
    by_origin: dict[str, list[int]] = {}
    for p, (i, _) in enumerate(pairs):
        by_origin.setdefault(i, []).append(p)
    gravity = np.zeros(len(pairs), dtype=np.float64)
    for origin, idxs in by_origin.items():
        weights = np.array(
            [max(populations.get(pairs[p][1], 0.0), 0.0) / (max(hops.get((origin, pairs[p][1]), 1), 1) ** distance_decay) for p in idxs],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0:
            continue
        emitted = base_rate * max(populations.get(origin, 0.0), 0.0)
        for k, p in enumerate(idxs):
            gravity[p] = emitted * weights[k] / total
    return gravity


def reconstruct_migration_flows_for_year(
    *,
    year: str,
    nodes: list[str],
    populations: dict[str, float],
    adjacency: dict[str, tuple[str, ...]],
    net_by_node: dict[str, float],
    hops: dict[tuple[str, str], int] | None = None,
    max_hops: int = 3,
    base_rate: float = 0.01,
    distance_decay: float = 2.0,
    net_weight: float = 10.0,
    gravity_weight: float = 1.0,
    census_flows: dict[tuple[str, str], float] | None = None,
    census_weight: float = 50.0,
    max_iterations: int = 4000,
) -> MigrationFlowReconstruction:
    """Reconstruct one year's O→D flow field. See module docstring for the model."""
    nodes = list(nodes)
    if len(nodes) < 2:
        raise MigrationFlowError("migration flow reconstruction needs >= 2 nodes")
    hops = hops if hops is not None else hop_distances(adjacency, nodes)
    pairs = _candidate_pairs(nodes, hops, max_hops)
    # Ensure any census-anchored pair is in the candidate support even if beyond max_hops.
    if census_flows:
        known = set(pairs)
        for (i, j) in census_flows:
            if i in populations and j in populations and i != j and (i, j) not in known:
                pairs.append((i, j))
                known.add((i, j))
    if not pairs:
        raise MigrationFlowError("no candidate migration pairs (graph disconnected or max_hops too small)")
    if len(pairs) > MAX_DENSE_FLOW_PAIRS:
        raise MigrationFlowError(
            f"candidate pair count {len(pairs)} exceeds dense reconstruction limit {MAX_DENSE_FLOW_PAIRS}; "
            "reduce max_hops or use a sparse backend"
        )

    node_index = {node: k for k, node in enumerate(nodes)}
    int_pairs = [(node_index[i], node_index[j]) for (i, j) in pairs]
    gravity = _gravity_prior(pairs, populations, hops, base_rate=base_rate, distance_decay=distance_decay)
    net = np.array([float(net_by_node.get(node, 0.0)) for node in nodes], dtype=np.float64)

    anchor_values = anchor_mask = None
    anchored = bool(census_flows)
    warnings: list[str] = []
    if census_flows:
        anchor_values = np.zeros(len(pairs), dtype=np.float64)
        anchor_mask = np.zeros(len(pairs), dtype=bool)
        pair_pos = {pair: p for p, pair in enumerate(pairs)}
        for pair, value in census_flows.items():
            p = pair_pos.get(pair)
            if p is not None:
                anchor_values[p] = max(float(value), 0.0)
                anchor_mask[p] = True
        # SCALE CALIBRATION: gross flow is unidentified from net alone (net is
        # invariant to balanced circulation) -- it is set entirely by the gravity
        # prior's scale (base_rate). When a census O->D anchor is present it carries
        # the *true* scale, so rescale the whole gravity prior to match the census
        # mass on the anchored pairs. Without this the prior's fixed base_rate fights
        # the anchor: anchored pairs get pulled to the (small) truth while unanchored
        # pairs stay at the (large) base_rate scale -- an inconsistency that degrades
        # the reconstruction. With it, the anchor's scale propagates to every pair.
        anchored_gravity = float(gravity[anchor_mask].sum())
        anchored_census = float(anchor_values[anchor_mask].sum())
        if anchored_gravity > 0 and anchored_census > 0:
            gravity = gravity * (anchored_census / anchored_gravity)
        else:
            warnings.append("migration_flow_census_scale_calibration_skipped")

    problem = migration_flow_instance(
        pairs=int_pairs,
        gravity_prior=gravity,
        net_by_node=net,
        net_weight=net_weight,
        gravity_weight=gravity_weight,
        census_anchor_values=anchor_values,
        census_anchor_mask=anchor_mask,
        census_weight=census_weight,
    )
    x0 = np.maximum(gravity, 0.0)
    solution, diagnostics = solve_ctr(problem, x0=x0, max_iters=max_iterations)

    flows = {pairs[p]: float(solution[p]) for p in range(len(pairs)) if solution[p] > 1e-9}
    S = net_flow_operator(len(nodes), int_pairs)
    net_residual_l1 = float(np.sum(np.abs(S @ solution - net)))
    if not diagnostics["converged"]:
        warnings.append("migration_flow_solver_nonconvergence")
    if not anchored:
        warnings.append("migration_flow_estimate_gravity_prior_no_census_od_anchor")
    return MigrationFlowReconstruction(
        year=str(year),
        node_ids=tuple(nodes),
        flows=flows,
        converged=bool(diagnostics["converged"]),
        net_residual_l1=net_residual_l1,
        gross_flow=float(solution.sum()),
        anchored=anchored,
        warnings=tuple(warnings),
    )


def reconstruct_migration_flows(
    *,
    nodes: list[str],
    populations_by_year: dict[str, dict[str, float]],
    net_by_year: dict[str, dict[str, float]],
    adjacency: dict[str, tuple[str, ...]],
    census_flows_by_year: dict[str, dict[tuple[str, str], float]] | None = None,
    max_hops: int = 3,
    base_rate: float = 0.01,
    distance_decay: float = 2.0,
    max_iterations: int = 4000,
) -> list[MigrationFlowReconstruction]:
    """Reconstruct the O→D flow field for every year that has a net marginal.

    ``populations_by_year[year][node]`` and ``net_by_year[year][node]`` come from the
    population tensor's closure totals and net-migration residual (MSD §2.8.7). The
    contiguity ``adjacency`` (shared across years) provides the candidate support and
    hop distances (computed once). Years with no net data are skipped.
    """
    # Bound the shared hop computation to the candidate radius: this is the national memory
    # path (all-pairs over ~5,570 nodes would be a multi-GB dict). Distances beyond max_hops are
    # never used by the candidate support or gravity prior for the non-anchored estimate.
    hops = hop_distances(adjacency, nodes, max_hops=max_hops)
    census_flows_by_year = census_flows_by_year or {}
    out: list[MigrationFlowReconstruction] = []
    for year in sorted(net_by_year):
        net = net_by_year[year]
        if not any(abs(v) > 0 for v in net.values()):
            continue
        out.append(
            reconstruct_migration_flows_for_year(
                year=year,
                nodes=nodes,
                populations=populations_by_year.get(year, {}),
                adjacency=adjacency,
                net_by_node=net,
                hops=hops,
                max_hops=max_hops,
                base_rate=base_rate,
                distance_decay=distance_decay,
                census_flows=census_flows_by_year.get(year),
                max_iterations=max_iterations,
            )
        )
    return out


__all__ = [
    "MigrationFlowError",
    "MigrationFlowReconstruction",
    "MAX_DENSE_FLOW_PAIRS",
    "hop_distances",
    "reconstruct_migration_flows_for_year",
    "reconstruct_migration_flows",
]
