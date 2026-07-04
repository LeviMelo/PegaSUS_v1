"""APC-01 — the Adaptive Precision Controller (MSD-III §V.5): uncertainty steers compute."""

from __future__ import annotations

from pegasus.compute.controller import Budget, Quantity, adaptive_precision_run


def test_escalates_numerically_dominated_leaves_data_limited() -> None:
    quantities = [
        # numerical error dominates and is cheap → escalate to exact → meets the target
        Quantity("edgeA", statistical_uncertainty=0.02, numerical_uncertainty=0.20, exact_cost=1.0),
        # data-limited (statistical ≫ numerical) → compute cannot help → never escalated
        Quantity("edgeB", statistical_uncertainty=0.30, numerical_uncertainty=0.05, exact_cost=1.0),
    ]
    report = adaptive_precision_run(quantities, target=0.1, budget=Budget(total=10.0))

    assert "edgeA" in report.met                         # numerical error removed → 0.02 ≤ 0.1
    assert "edgeB" in report.data_limited                # 0.30 > 0.1 but compute is useless here
    assert report.quantities["edgeB"].numerical_uncertainty == 0.05   # budget not wasted on it
    assert report.spent == 1.0                           # only edgeA escalated


def test_reports_approximation_limited_when_budget_exhausted() -> None:
    quantities = [Quantity("expensive", 0.02, 0.5, exact_cost=100.0)]
    report = adaptive_precision_run(quantities, target=0.1, budget=Budget(total=10.0))

    assert "expensive" in report.approximation_limited   # unmet + numerical-dominated, unaffordable
    assert report.spent == 0.0                           # nothing silently spent


def test_value_of_computation_prioritizes_highest_reduction_per_cost() -> None:
    quantities = [
        Quantity("high_voc", 0.01, 0.40, exact_cost=1.0),   # more numerical error per unit cost
        Quantity("low_voc", 0.01, 0.30, exact_cost=1.0),
    ]
    report = adaptive_precision_run(quantities, target=0.05, budget=Budget(total=1.0))  # affords one

    assert "high_voc" in report.met
    assert "low_voc" in report.approximation_limited
