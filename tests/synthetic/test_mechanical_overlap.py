"""DIS-05 — the shared-code mechanical-overlap guard (MSD-III §II.6/§5.3).

Concept-variables built on overlapping code sets are *mechanically* correlated (they
count overlapping events). Such an edge must be typed ``mechanical_overlap`` and never
promoted as an epidemiological discovery.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.edges import type_mechanical_overlap
from pegasus.ldo.orchestrator import run_ldo
from pegasus.ldo.records import LinkRecord


def test_high_overlap_edge_typed_mechanical() -> None:
    records = [
        LinkRecord("A", "B", "contemporaneous", weight=0.9, certification_status="selected"),
        LinkRecord("A", "C", "contemporaneous", weight=0.3, certification_status="selected"),
    ]
    code_sets = {
        "A": frozenset({"I21", "I22", "I23", "I24"}),
        "B": frozenset({"I21", "I22", "I23", "I25"}),   # ~60% Jaccard with A → mechanical
        "C": frozenset({"J18", "J12"}),                 # disjoint from A → genuine
    }
    typed = type_mechanical_overlap(records, code_sets, threshold=0.5)
    ab = next(r for r in typed if r.target_var == "B")
    ac = next(r for r in typed if r.target_var == "C")

    assert ab.edge_type == "mechanical_overlap"
    assert ab.overlap_jaccard is not None and ab.overlap_jaccard >= 0.5
    assert ab.certification_status == "descriptive"          # never a promoted discovery
    assert "mechanical_overlap_shared_codes" in ab.warnings
    # the disjoint pair keeps its type but is annotated with its (zero) overlap
    assert ac.edge_type == "contemporaneous"
    assert ac.overlap_jaccard == 0.0


def test_run_ldo_does_not_report_overlap_as_discovery() -> None:
    """Two variables sharing codes co-vary; run_ldo must NOT certify that as a real edge."""
    rng = np.random.default_rng(0)
    S, T = 14, 24
    shared = rng.standard_normal((S, T))                     # A and B both track a shared signal
    X = np.stack([shared + 0.1 * rng.standard_normal((S, T)),
                  shared + 0.1 * rng.standard_normal((S, T)),
                  rng.standard_normal((S, T))], axis=0)
    field = LDOField(
        variables=("A", "B", "C"),
        space_ids=tuple(str(270000 + i) for i in range(S)),
        time_ids=tuple(range(T)),
        X=X, W=np.ones_like(X), resolution="month",
    )
    meta = {
        "A": {"code_set": frozenset({"I21", "I22", "I23", "I24"}), "code_system": "CID-10",
              "topology_role": "underlying_cause", "projection_status": "exact"},
        "B": {"code_set": frozenset({"I21", "I22", "I23", "I25"}), "code_system": "CID-10",
              "topology_role": "underlying_cause", "projection_status": "exact"},
        "C": {"code_set": frozenset({"J18"}), "code_system": "CID-10",
              "topology_role": "underlying_cause", "projection_status": "exact"},
    }
    run = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0, variable_meta=meta)

    ab = [r for r in run.link_records if {r.source_var, r.target_var} == {"A", "B"}]
    assert ab, "the strong A-B co-movement should surface as an edge"
    for r in ab:
        # it exists, but as known shared-code structure — never a certified discovery
        assert r.edge_type == "mechanical_overlap"
        assert r.certification_status != "selected"
        assert r.code_system == "CID-10" and r.topology_role == "underlying_cause"
