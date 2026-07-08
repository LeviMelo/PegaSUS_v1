"""Dependence-aware FDR is live in run_ldo, enforced + Benjamini-Yekutieli by default (Theme-5/W4).

The unit mechanics live in test_ldo_fdr.py; this pins the ORCHESTRATOR seam that no other test
covers: the default run annotates fdr_qvalue/fdr_method (BY) on the real records AND enforces the
gate, while a genuinely-null selected edge injected at the live certify path falls to descriptive
under the default and is left alone only when the caller opts back to annotate-only.
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


def test_run_ldo_default_enforces_dependence_robust_fdr_on_live_records():
    field = _field()
    base = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0)

    # the enforced dependence-robust FDR reaches the live records: q + BY method on testable edges.
    testable = [r for r in base.link_records if r.n_eff is not None and r.partial_correlation is not None]
    assert testable, "expected at least one testable edge in the run"
    assert all(r.fdr_qvalue is not None for r in testable)
    assert all(r.fdr_method == "benjamini_yekutieli_fisherz_neff" for r in testable)

    # the default now enforces, so an explicit enforce=True policy is decision-identical to it,
    # and passing the default-constructed policy is byte-identical to the None default.
    enforced = run_ldo(
        field, K=1, n_subsamples=6, run_residual_scan=False, seed=0,
        certification_policy=LDOCertificationPolicy(fdr_enforce=True),
    )
    key = lambda r: (r.edge_type, r.source_var, r.target_var, r.lag_k)
    b = {key(r): r for r in base.link_records}
    e = {key(r): r for r in enforced.link_records}
    assert set(b) == set(e)
    assert all(b[k].certification_status == e[k].certification_status for k in b)
    ann = run_ldo(
        field, K=1, n_subsamples=6, run_residual_scan=False, seed=0,
        certification_policy=LDOCertificationPolicy(),
    )
    assert [r.as_row() for r in base.link_records] == [r.as_row() for r in ann.link_records]

    # the genuinely-strong A–B edge survives enforcement — real signal is not suppressed.
    ab = [r for r in base.link_records if {r.source_var, r.target_var} == {"A", "B"}]
    assert ab and any(r.certification_status == "selected" for r in ab)


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

    # opt-out annotate-only leaves the null edge selected; the enforced DEFAULT falls it descriptive.
    annotated = {(r.source_var, r.target_var): r
                 for r in certify_links(records, policy=LDOCertificationPolicy(fdr_enforce=False))}
    enforced = {(r.source_var, r.target_var): r for r in certify_links(records)}

    assert annotated[("Z", "Y")].certification_status == "selected"     # opt-out: unchanged
    assert enforced[("Z", "Y")].certification_status == "descriptive"   # default enforce: null falls
    assert "fdr_not_significant_descriptive_only" in enforced[("Z", "Y")].warnings
    # the genuinely-strong A–B edge survives enforcement.
    assert enforced[("A", "B")].certification_status == "selected"
