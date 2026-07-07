"""Dependence-aware FDR is live in run_ldo, annotate-only by default (Theme-5).

The unit mechanics live in test_ldo_fdr.py; this pins the ORCHESTRATOR seam that no other
test covers: the default run annotates fdr_qvalue/fdr_method on the real records without
moving any certification decision, and enforce reaches the live certify path and downgrades a
genuinely-null selected edge to descriptive.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.certify import LDOCertificationPolicy, certify_links
from pegasus.ldo.orchestrator import run_ldo
from pegasus.ldo.records import LinkRecord


def _field() -> LDOField:
    rng = np.random.default_rng(0)
    S, T = 80, 12
    A = rng.standard_normal((S, T))
    B = 0.9 * A + 0.3 * rng.standard_normal((S, T))  # strong, genuinely-significant A–B
    noise = [rng.standard_normal((S, T)) for _ in range(6)]
    X = np.stack([A, B, *noise], axis=0)
    variables = ("A", "B") + tuple(f"n{i}" for i in range(6))
    return LDOField(
        variables=variables,
        space_ids=tuple(str(270000 + i) for i in range(S)),
        time_ids=tuple(range(T)),
        X=X,
        W=np.ones_like(X),
        resolution="year",
    )


def test_run_ldo_default_annotates_fdr_without_moving_certification():
    field = _field()
    base = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0)

    # annotate-only default reaches the live records: q + method populated on testable edges.
    testable = [r for r in base.link_records if r.n_eff is not None and r.partial_correlation is not None]
    assert testable, "expected at least one testable edge in the run"
    assert all(r.fdr_qvalue is not None for r in testable)
    assert all(r.fdr_method == "benjamini_hochberg_fisherz_neff" for r in testable)

    # enforcing FDR through the orchestrator does NOT move any decision when every selected edge
    # is genuinely significant — the annotate-only default is decision-identical to enforce here.
    enforced = run_ldo(
        field, K=1, n_subsamples=6, run_residual_scan=False, seed=0,
        certification_policy=LDOCertificationPolicy(fdr_enforce=True),
    )
    key = lambda r: (r.edge_type, r.source_var, r.target_var, r.lag_k)
    b = {key(r): r for r in base.link_records}
    e = {key(r): r for r in enforced.link_records}
    assert set(b) == set(e)
    assert all(b[k].certification_status == e[k].certification_status for k in b)

    # passing an explicit annotate-only policy is byte-identical to the None default.
    ann = run_ldo(
        field, K=1, n_subsamples=6, run_residual_scan=False, seed=0,
        certification_policy=LDOCertificationPolicy(),
    )
    assert [r.as_row() for r in base.link_records] == [r.as_row() for r in ann.link_records]


def test_enforce_downgrades_a_null_selected_edge_on_the_live_record_type():
    # Direction proof at the live certify path: inject one genuinely-null but pre-selected edge
    # alongside the real run's records; enforce must fall it to descriptive, default must not.
    base = run_ldo(_field(), K=1, n_subsamples=6, run_residual_scan=False, seed=0)
    null_edge = LinkRecord(
        source_var="Z", target_var="Y", edge_type="contemporaneous",
        weight=0.01, partial_correlation=0.01, stability=0.9, uncertainty=0.05,
        n_eff=900.0, certification_status="selected",
    )
    records = list(base.link_records) + [null_edge]
    # strip stale annotations so certify recomputes q over the combined set
    records = [replace(r, fdr_qvalue=None, fdr_method=None) for r in records]

    default = {(r.source_var, r.target_var): r for r in certify_links(records)}
    enforced = {(r.source_var, r.target_var): r
                for r in certify_links(records, policy=LDOCertificationPolicy(fdr_enforce=True))}

    assert default[("Z", "Y")].certification_status == "selected"       # annotate-only: unchanged
    assert enforced[("Z", "Y")].certification_status == "descriptive"   # enforce: null edge falls
    assert "fdr_not_significant_descriptive_only" in enforced[("Z", "Y")].warnings
    # the genuinely-strong A–B edge survives enforcement in both.
    assert enforced[("A", "B")].certification_status == "selected"
