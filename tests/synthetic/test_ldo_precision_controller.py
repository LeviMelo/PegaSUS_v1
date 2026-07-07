"""WP7 / §V.5 — the Adaptive Precision Controller, wired into run_ldo.

Uncertainty drives compute: when the reducible NUMERICAL (randomized-SVD) error dominates a
decision-relevant quantity, escalate it to exact; leave data-limited quantities alone (compute
cannot help them). Previously the controller (compute/controller.py) had zero callers on the
run_ldo path. This pins the escalate/leave-alone logic and that run_ldo now emits its verdict.
"""

from __future__ import annotations

import numpy as np

from pegasus.compute.controller import Budget, Quantity, adaptive_precision_run
from pegasus.ldo.assemble import LDOField
from pegasus.ldo.orchestrator import run_ldo


def test_controller_escalates_numerically_dominated_and_spares_data_limited():
    # numerically dominated + affordable → escalated to exact (numerical → 0), target met
    num_dom = Quantity(name="num", statistical_uncertainty=0.01, numerical_uncertainty=0.2, exact_cost=1.0)
    # data-limited (statistical dominates) → escalation would waste budget → left alone
    data_lim = Quantity(name="dat", statistical_uncertainty=0.3, numerical_uncertainty=0.01, exact_cost=1.0)
    report = adaptive_precision_run([num_dom, data_lim], target=0.05, budget=Budget(total=5.0))
    assert "num" in report.met and report.spent >= 1.0        # escalated
    assert "dat" in report.data_limited                        # not escalated (compute can't help)


def test_controller_respects_budget():
    q = Quantity(name="q", statistical_uncertainty=0.01, numerical_uncertainty=0.5, exact_cost=10.0)
    report = adaptive_precision_run([q], target=0.05, budget=Budget(total=1.0))  # too poor to escalate
    assert report.spent == 0.0 and "q" in report.approximation_limited


def _field():
    rng = np.random.default_rng(0)
    p, S, T = 4, 40, 6
    X = rng.standard_normal((p, S, T))
    X[1] += 0.7 * X[0]
    return LDOField(variables=tuple("ABCD"), space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                    time_ids=tuple(range(T)), X=X, W=np.ones((p, S, T)), resolution="year")


def test_run_ldo_emits_controller_verdict_when_targeted():
    run = run_ldo(_field(), K=2, n_subsamples=4, run_residual_scan=False, seed=0,
                  precision_target=0.01, precision_budget=1.0)
    pc = run.diagnostics["precision_controller"]
    assert pc is not None
    assert set(pc) >= {"target", "spent", "met", "approximation_limited", "data_limited", "escalated_to_exact"}
    assert isinstance(pc["escalated_to_exact"], bool)


def test_controller_off_by_default_is_no_op():
    run = run_ldo(_field(), K=2, n_subsamples=4, run_residual_scan=False, seed=0)
    assert run.diagnostics["precision_controller"] is None
