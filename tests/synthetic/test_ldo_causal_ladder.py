"""WP5 / §IV — the typed causal ladder: collider (Rung-1) + ITS escalation (Rung-2).

Rung-0 = LDO associational/temporal baseline; Rung-1 = orientation without experiments
(pairwise non-Gaussian LiNGAM + collider/v-structure); Rung-2 = quasi-experimental
(interrupted time series). Rung-3 (do-calculus) is expert-invoked only — never autonomous.
This pins that colliders are actually oriented (the previously-orphaned `is_collider`) and
that a directed edge with a structural break in its target escalates to Rung 2.
"""

from __future__ import annotations

import numpy as np

from pegasus.causal.orient import orient_links
from pegasus.causal.quasi import escalate_rung2_its
from pegasus.ldo.records import LinkRecord


def test_collider_v_structure_is_oriented_rung1():
    rng = np.random.default_rng(0)
    n = 4000
    # A, B independent and non-Gaussian; C is a common EFFECT (A→C←B): a v-structure.
    a = rng.exponential(size=n) - 1.0
    b = rng.exponential(size=n) - 1.0
    c = a + b + 0.1 * rng.standard_normal(n)
    data = {"A": a, "B": b, "C": c}
    # unshielded triple A–C–B (A,B not linked to each other)
    records = [
        LinkRecord(source_var="A", target_var="C", edge_type="contemporaneous"),
        LinkRecord(source_var="B", target_var="C", edge_type="contemporaneous"),
    ]
    oriented = orient_links(records, data)
    # both edges point INTO the collider C, tagged Rung-1 collider
    assert all(r.target_var == "C" for r in oriented), [(r.source_var, r.target_var) for r in oriented]
    assert all(r.causal_rung == 1 for r in oriented)
    assert all("collider_v_structure" in r.causal_assumptions for r in oriented)
    assert all("oriented_collider" in r.warnings for r in oriented)


def test_rung0_stays_when_unidentifiable_and_ladder_never_downgrades():
    rng = np.random.default_rng(1)
    n = 3000
    # jointly Gaussian, no v-structure → direction unidentifiable → Rung-0 associational
    x = rng.standard_normal(n)
    y = 0.6 * x + rng.standard_normal(n)
    records = [LinkRecord(source_var="X", target_var="Y", edge_type="contemporaneous")]
    oriented = orient_links(records, {"X": x, "Y": y})
    assert oriented[0].causal_rung == 0
    assert "orientation_undirected_unidentifiable" in oriented[0].warnings


def test_rung2_its_escalation_on_directed_edge_with_structural_break():
    T = 60
    # a directed (Rung-1) edge whose TARGET series has a sharp level shift at t=30
    series = np.concatenate([np.zeros(30), np.ones(30) * 5.0]) + 0.05 * np.arange(T)
    directed = LinkRecord(source_var="S", target_var="Y", edge_type="lagged_directed",
                          lag_k=3, causal_rung=1, causal_assumptions=("time_precedence",))
    undirected = LinkRecord(source_var="P", target_var="Q", edge_type="contemporaneous", causal_rung=0)
    out = escalate_rung2_its([directed, undirected], {"Y": series, "Q": np.zeros(T)})
    y_edge = next(r for r in out if r.target_var == "Y")
    q_edge = next(r for r in out if r.target_var == "Q")
    assert y_edge.causal_rung == 2  # escalated: break detected + ITS significant
    assert "interrupted_time_series" in y_edge.causal_assumptions
    assert any(w.startswith("rung2_its_shock_t") for w in y_edge.warnings)
    assert q_edge.causal_rung == 0  # a Rung-0 edge is never auto-escalated (ladder goes up only)
