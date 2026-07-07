"""WP2 / §VIII.2 — bounded-exhaustiveness in the coarse→fine multiresolution scan.

The coarse pass must be a recall-tuned FILTER on SENSITIVITY (§VIII.2(1)) — subgroup
heterogeneity / max-subgroup — not the aggregate-mean test. The decisive case is a CANCELLATION
edge: opposite-sign effects across spatial units sum to a pooled ≈ 0, so a pooled-mean screen
prunes it (a false negative), while the subgroup screen keeps it. A random DEEP AUDIT of pruned
variables (§VIII.2(2)) then measures the false-negative rate, and a typed coverage manifest
(§VIII.2(3)) records searched vs unsearched pairs.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.margins import GaussianField
from pegasus.ldo.records import LinkRecord
from pegasus.ldo.orchestrator import LDORun
from pegasus.ldo.resolution import _sensitivity_screen, run_multiresolution_ldo


def _cancellation_field(seed: int = 0) -> GaussianField:
    """A~B correlate +1 in half the spatial units and −1 in the other half → pooled corr ≈ 0
    (a cancellation edge). C is pure noise. The subgroup screen must drill A~B, not A~C."""
    rng = np.random.default_rng(seed)
    S, T = 6, 200   # long series → per-unit noise correlations stay well below the drill thresholds
    A = rng.standard_normal((S, T))
    B = np.empty((S, T))
    for s in range(S):
        sign = 1.0 if s < S // 2 else -1.0        # opposite-sign association across blocks
        B[s] = sign * A[s] + 0.1 * rng.standard_normal(T)
    C = rng.standard_normal((S, T))
    Z = np.stack([A, B, C])                        # (3, S, T)
    return GaussianField(variables=("A", "B", "C"),
                         space_ids=tuple(f"{27 + s}{s:04d}"[:6] for s in range(S)),
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((3, S, T)), resolution="year")


def test_subgroup_screen_catches_cancellation_edge_the_pooled_mean_hides():
    field = _cancellation_field()
    # empty coarse records → the pooled limb contributes nothing; only the subgroup limb can fire.
    run = LDORun(link_records=[], variables=field.variables, diagnostics={})
    pairs, reasons = _sensitivity_screen(run, field)
    kept = set(pairs)
    assert ("A", "B") in kept                       # cancellation edge drilled by heterogeneity
    assert reasons[("A", "B")].startswith("subgroup:")
    assert ("A", "C") not in kept and ("B", "C") not in kept   # pure noise pruned


def test_pooled_signal_limb_still_keeps_a_recurrent_edge():
    field = _cancellation_field()
    records = [LinkRecord(source_var="A", target_var="C", edge_type="contemporaneous",
                          weight=0.02, stability=0.4, certification_status="descriptive")]
    run = LDORun(link_records=records, variables=field.variables, diagnostics={})
    pairs, reasons = _sensitivity_screen(run, field)
    assert ("A", "C") in set(pairs)                 # kept via the pooled stability-recurrence limb


def _field(seed: int = 0) -> GaussianField:
    rng = np.random.default_rng(seed)
    p, S, T = 8, 40, 6
    Z = rng.standard_normal((p, S, T))
    Z[1] += 0.8 * Z[0]            # a real strong edge (candidate)
    return GaussianField(variables=tuple("ABCDEFGH"),
                         space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((p, S, T)), resolution="year")


def test_multiresolution_run_reports_screen_audit_and_coverage_manifest():
    run = run_multiresolution_ldo(_field(), _field(), coarse_K=2, fine_K=2,
                                  n_subsamples=4, seed=0)
    d = run.diagnostics
    assert d["sensitivity_screen"] == "subgroup_heterogeneity_or_max_or_pooled"
    # §VIII.2(3) typed coverage manifest is emitted with a STATED (string) sparsity assumption
    cm = run.coverage_manifest
    assert cm is not None
    assert isinstance(cm.sparsity_of_truth_assumption, str) and cm.sparsity_of_truth_assumption
    assert (len(cm.searched) + len(cm.unsearched)) >= 1
    if d.get("candidates"):  # §VIII.2(2) random deep audit exercised when a candidate was found
        audit = d["random_deep_audit"]
        assert "false_negative_rate" in audit and "n_audit_vars" in audit
