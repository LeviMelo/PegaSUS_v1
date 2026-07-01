"""Latent O→D migration-flow reconstruction + migration-affinity spatial kernel
(MSD §2.8.7 flow layer / MSD-II §II.4 context_derived graph).

Migration is inferred from net-migration marginals by a gravity-structured CTR
reconstruction (structure), pinned to census O→D flows when available (scale +
bilateral truth). The reconstructed flows induce a functional adjacency kernel.
"""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pegasus.geo.migration_affinity import build_migration_affinity_graph
from pegasus.geo.spatial_graph import SpatialCircularityError, assert_spatial_legality
from pegasus.she.reconstruction.instances import migration_flow_instance, net_flow_operator
from pegasus.she.reconstruction.problem import evaluate_ctr, solve_ctr
from pegasus.sidra.facts import normalize_flat_records_to_facts, write_facts_parquet
from pegasus.sidra.population_cube.build import solve_population_tensor_from_sidra_strata
from pegasus.sidra.population_cube.migration import (
    MigrationFlowError,
    hop_distances,
    reconstruct_migration_flows_for_year,
)


def _grid(g: int = 4):
    nodes = [f"{r}{c}" for r in range(g) for c in range(g)]

    def nbr(r, c):
        return [f"{r+dr}{c+dc}" for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if 0 <= r + dr < g and 0 <= c + dc < g]

    adjacency = {f"{r}{c}": tuple(nbr(r, c)) for r in range(g) for c in range(g)}
    return nodes, adjacency


def _true_gravity_flows(nodes, adjacency, populations, hops, max_hops=2, rate=0.02, rng=None):
    rng = rng or np.random.default_rng(7)
    total_pop = sum(populations.values())
    flows = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            d = hops.get((i, j))
            if d and d <= max_hops:
                base = rate * populations[i] * populations[j] / (d ** 2) / total_pop
                flows[(i, j)] = base * (1.0 + 0.3 * rng.random())
    return flows


def test_hop_distances_bfs():
    nodes, adjacency = _grid(3)
    hops = hop_distances(adjacency, nodes)
    assert hops[("00", "00")] == 0
    assert hops[("00", "01")] == 1
    assert hops[("00", "22")] == 4  # Manhattan path on a rook grid


def test_net_flow_operator_is_the_balancing_identity():
    # 3 nodes, flows 0->1 (5) and 1->2 (3): net = in - out.
    pairs = [(0, 1), (1, 2)]
    S = net_flow_operator(3, pairs)
    F = np.array([5.0, 3.0])
    net = S @ F
    assert list(net) == [-5.0, 2.0, 3.0]  # node0: -5 out; node1: +5-3; node2: +3


def test_migration_flow_instance_gradient_is_exact():
    pairs = [(0, 1), (1, 0), (1, 2)]
    prob = migration_flow_instance(
        pairs=pairs, gravity_prior=[2.0, 1.0, 3.0], net_by_node=[0.5, -0.2, 0.1],
        net_weight=4.0, gravity_weight=1.0,
    )
    x = np.array([1.3, 0.7, 2.1])
    ev = evaluate_ctr(prob, x)
    num = np.zeros(3)
    eps = 1e-6
    for i in range(3):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        num[i] = (evaluate_ctr(prob, xp).loss - evaluate_ctr(prob, xm).loss) / (2 * eps)
    assert np.max(np.abs(ev.gradient - num)) < 1e-5


def test_reconstruction_recovers_gravity_structure_from_net_alone():
    nodes, adjacency = _grid(4)
    rng = np.random.default_rng(3)
    populations = {n: float(rng.integers(5000, 50000)) for n in nodes}
    hops = hop_distances(adjacency, nodes)
    true_flows = _true_gravity_flows(nodes, adjacency, populations, hops, rng=rng)
    node_index = {n: k for k, n in enumerate(nodes)}
    pairs = list(true_flows)
    S = net_flow_operator(len(nodes), [(node_index[i], node_index[j]) for (i, j) in pairs])
    net_vec = S @ np.array([true_flows[p] for p in pairs])
    net_by_node = {nodes[k]: float(net_vec[k]) for k in range(len(nodes))}

    rec = reconstruct_migration_flows_for_year(
        year="2016", nodes=nodes, populations=populations, adjacency=adjacency,
        net_by_node=net_by_node, max_hops=2, base_rate=0.02,
    )
    recon = np.array([rec.flows.get(p, 0.0) for p in pairs])
    truth = np.array([true_flows[p] for p in pairs])
    # Structure (relative allocation) is recovered even from net marginals alone.
    assert float(np.corrcoef(recon, truth)[0, 1]) > 0.85
    assert "migration_flow_estimate_gravity_prior_no_census_od_anchor" in rec.warnings


def test_census_anchor_calibrates_scale_and_improves_recovery():
    nodes, adjacency = _grid(4)
    rng = np.random.default_rng(5)
    populations = {n: float(rng.integers(5000, 50000)) for n in nodes}
    hops = hop_distances(adjacency, nodes)
    true_flows = _true_gravity_flows(nodes, adjacency, populations, hops, rng=rng)
    node_index = {n: k for k, n in enumerate(nodes)}
    pairs = list(true_flows)
    S = net_flow_operator(len(nodes), [(node_index[i], node_index[j]) for (i, j) in pairs])
    net_vec = S @ np.array([true_flows[p] for p in pairs])
    net_by_node = {nodes[k]: float(net_vec[k]) for k in range(len(nodes))}

    def corr(rec):
        return float(np.corrcoef(np.array([rec.flows.get(p, 0.0) for p in pairs]), np.array([true_flows[p] for p in pairs]))[0, 1])

    no_anchor = reconstruct_migration_flows_for_year(
        year="2016", nodes=nodes, populations=populations, adjacency=adjacency,
        net_by_node=net_by_node, max_hops=2, base_rate=0.02,
    )
    half = {p: true_flows[p] for p in pairs[::2]}
    with_anchor = reconstruct_migration_flows_for_year(
        year="2016", nodes=nodes, populations=populations, adjacency=adjacency,
        net_by_node=net_by_node, max_hops=2, base_rate=0.02, census_flows=half, census_weight=100.0,
    )
    assert with_anchor.anchored and not no_anchor.anchored
    assert corr(with_anchor) >= corr(no_anchor)
    assert with_anchor.net_residual_l1 <= no_anchor.net_residual_l1 + 1e-6


def test_pair_count_guard_refuses_oversized_dense_problem():
    nodes, adjacency = _grid(4)
    populations = {n: 10000.0 for n in nodes}
    net = {n: 0.0 for n in nodes}
    net["00"] = 5.0
    net["33"] = -5.0
    with pytest.raises(MigrationFlowError):
        # max_hops huge => fully-connected candidate set; monkeypatch the cap low.
        import pegasus.sidra.population_cube.migration as m
        old = m.MAX_DENSE_FLOW_PAIRS
        m.MAX_DENSE_FLOW_PAIRS = 10
        try:
            reconstruct_migration_flows_for_year(
                year="2016", nodes=nodes, populations=populations, adjacency=adjacency,
                net_by_node=net, max_hops=6,
            )
        finally:
            m.MAX_DENSE_FLOW_PAIRS = old


def test_affinity_kernel_is_symmetric_context_derived_and_guarded():
    nodes, adjacency = _grid(4)
    rng = np.random.default_rng(11)
    populations = {n: float(rng.integers(5000, 50000)) for n in nodes}
    hops = hop_distances(adjacency, nodes)
    true_flows = _true_gravity_flows(nodes, adjacency, populations, hops, rng=rng)
    node_index = {n: k for k, n in enumerate(nodes)}
    pairs = list(true_flows)
    S = net_flow_operator(len(nodes), [(node_index[i], node_index[j]) for (i, j) in pairs])
    net_vec = S @ np.array([true_flows[p] for p in pairs])
    net_by_node = {nodes[k]: float(net_vec[k]) for k in range(len(nodes))}
    rec = reconstruct_migration_flows_for_year(
        year="2016", nodes=nodes, populations=populations, adjacency=adjacency,
        net_by_node=net_by_node, max_hops=2, base_rate=0.02,
    )
    graph = build_migration_affinity_graph([rec], populations, top_k=4)
    assert graph.legality_class == "context_derived"
    assert graph.weighted
    assert np.allclose(graph.view("symmetric"), graph.view("symmetric").T)
    assert np.allclose(graph.view("laplacian").sum(axis=1), 0.0, atol=1e-9)
    assert_spatial_legality(graph, ["gdp", "sanitation"])  # provenance-disjoint: legal
    with pytest.raises(SpatialCircularityError):
        assert_spatial_legality(graph, ["population"])  # shares provenance: illegal


# --- build integration on real contiguous municipalities -------------------
MUNIS = ["270010", "270240", "270330", "270500", "270580", "270642"]


def _pop_records(table, var, year, per_muni):
    return [
        {"table_id": table, "variable_id": var, "period": year, "locality_level": "N6",
         "locality_id": m + "0", "classification_tuple": [], "category_tuple": [], "value": str(v), "unit": "P"}
        for m, v in per_muni.items()
    ]


def test_build_emits_migration_flow_and_affinity_artifacts(tmp_path: Path):
    # Degenerate strata (total-only) at 2015 + annual 6579 totals so the tensor has
    # a real per-year closure, plus civil-registry births/deaths for the residual.
    base = {m: 20000 + 1000 * i for i, m in enumerate(MUNIS)}
    strata = _pop_records("9606", "93", "2015", base)
    strata_path = tmp_path / "strata.parquet"
    write_facts_parquet(normalize_flat_records_to_facts(strata, table_id="9606", request_hash="r", metadata_hash="m"), output_path=strata_path)

    totals = list(strata)
    for y, delta in (("2016", 300), ("2017", 260)):
        totals += _pop_records("6579", "9324", y, {m: base[m] + delta * (i + 1) for i, m in enumerate(MUNIS)})
    totals_path = tmp_path / "totals.parquet"
    write_facts_parquet(normalize_flat_records_to_facts(totals, table_id="mix", request_hash="r", metadata_hash="m"), output_path=totals_path)

    births = []
    deaths = []
    for y in ("2016", "2017"):
        births += _pop_records("2609", "217", y, {m: 200 for m in MUNIS})
        deaths += _pop_records("2683", "343", y, {m: 150 for m in MUNIS})
    births_path, deaths_path = tmp_path / "b.parquet", tmp_path / "d.parquet"
    write_facts_parquet(normalize_flat_records_to_facts(births, table_id="2609", request_hash="r", metadata_hash="m"), output_path=births_path)
    write_facts_parquet(normalize_flat_records_to_facts(deaths, table_id="2683", request_hash="r", metadata_hash="m"), output_path=deaths_path)

    build = solve_population_tensor_from_sidra_strata(
        population_strata_path=strata_path, total_anchor_path=totals_path,
        output_path=tmp_path / "tensor.parquet", mode="independent_denominator",
        civil_registry_births_path=births_path, civil_registry_deaths_path=deaths_path,
        reconstruct_migration=True, migration_max_hops=2, max_iterations=2000,
    )
    manifest = build.as_manifest()
    assert build.migration_flows_path is not None and Path(build.migration_flows_path).exists()
    assert build.migration_affinity_path is not None and Path(build.migration_affinity_path).exists()
    assert len(manifest["migration_flow_reconstructions"]) >= 1

    flows = pl.read_parquet(build.migration_flows_path)
    assert set(flows.columns) == {"year", "origin_cod6", "destination_cod6", "flow"}
    assert (flows["flow"] > 0).all()
    # every origin/destination is one of the run's municipalities
    assert set(flows["origin_cod6"].to_list()) <= set(MUNIS)

    affinity = pl.read_parquet(build.migration_affinity_path)
    assert set(affinity.columns) == {"source_cod6", "target_cod6", "affinity"}
    assert (affinity["affinity"] > 0).all()
