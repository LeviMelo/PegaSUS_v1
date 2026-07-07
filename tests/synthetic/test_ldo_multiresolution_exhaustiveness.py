"""WP2 / §VIII.2 — bounded-exhaustiveness in the coarse→fine multiresolution scan.

The coarse pass must be a recall-tuned FILTER (screen on sensitivity, §VIII.2(1)), not the
aggregate-certified TEST — else a sub-threshold-but-real edge is silently pruned. And a random
DEEP AUDIT of pruned variables (§VIII.2(2)) measures the false-negative rate empirically, so
"no edge" is a measured claim, not an assumption.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.margins import GaussianField
from pegasus.ldo.records import LinkRecord
from pegasus.ldo.orchestrator import LDORun
from pegasus.ldo.resolution import _sensitivity_screen, run_multiresolution_ldo


def test_sensitivity_screen_keeps_subthreshold_signal_the_test_would_drop():
    records = [
        # certified — kept by both the test and the screen
        LinkRecord(source_var="A", target_var="B", edge_type="contemporaneous",
                   weight=0.4, stability=0.9, certification_status="selected"),
        # NOT certified but a real sub-threshold signal (stability recurrence) — the aggregate
        # test drops it; the sensitivity screen keeps it as a candidate to refine.
        LinkRecord(source_var="C", target_var="D", edge_type="contemporaneous",
                   weight=0.08, stability=0.3, certification_status="descriptive"),
        # pure noise (no weight, no stability) — screened out
        LinkRecord(source_var="E", target_var="F", edge_type="contemporaneous",
                   weight=0.0, stability=None, certification_status="descriptive"),
    ]
    run = LDORun(link_records=records, variables=("A", "B", "C", "D", "E", "F"), diagnostics={})
    kept = set(_sensitivity_screen(run, sensitivity_threshold=0.05))
    assert ("A", "B") in kept and ("C", "D") in kept  # certified AND sub-threshold-but-real
    assert ("E", "F") not in kept                       # pure noise pruned


def _field(seed: int = 0) -> GaussianField:
    rng = np.random.default_rng(seed)
    p, S, T = 8, 40, 6
    Z = rng.standard_normal((p, S, T))
    Z[1] += 0.8 * Z[0]            # a real strong edge (candidate)
    return GaussianField(variables=tuple("ABCDEFGH"),
                         space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((p, S, T)), resolution="year")


def test_multiresolution_run_reports_screen_and_random_deep_audit():
    run = run_multiresolution_ldo(_field(), _field(), coarse_K=2, fine_K=2,
                                  n_subsamples=4, seed=0)
    d = run.diagnostics
    if d.get("candidates"):  # a candidate was found → the §VIII.2 machinery is exercised
        assert d["sensitivity_screen"] == "recall_tuned_signal_or_stability"
        audit = d["random_deep_audit"]
        assert "false_negative_rate" in audit and "n_audit_vars" in audit
