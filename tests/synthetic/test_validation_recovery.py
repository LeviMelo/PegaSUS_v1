"""VAL-03 — synthetic ground-truth recovery (MSD-III §IX.2).

The standing separability/recovery prong: plant a known structure (two directed-lagged
edges + a shared latent factor + noise variables), run the real LDO, and score recovery.
A certified engine recovers the planted edges, attributes the shared factor as
``latent_shared`` (not a web of direct edges), and gets the lag right — the empirical
test of separability adequacy that argument cannot settle.
"""

from __future__ import annotations

from pegasus.ldo.orchestrator import run_ldo
from pegasus.validation.synthetic import (
    PlantedEdge,
    PlantedTruth,
    generate_planted_field,
    recovery_score,
)


def test_ldo_recovers_planted_structure() -> None:
    # The shared factor loads E, F AND G. A latent driver must span >= 3 variables to be
    # *identifiable*: a rank-1 component on only 2 variables is observationally identical to
    # a direct edge between them (Ω = S − L is non-unique there — the CPW incoherence
    # condition), so a 2-variable "factor" is honestly reported as a direct edge, not a
    # confounder. A 3-variable factor is distinguishable from a direct-edge triangle and is
    # the case where "confounded, not directly linked" is a real, testable claim.
    truth = PlantedTruth(
        variables=("A", "B", "C", "D", "E", "F", "G", "H"),
        edges=(PlantedEdge("A", "B", 6, 1.0), PlantedEdge("C", "D", 0, 1.0)),
        factors=((("E", "F", "G"), 1.2),),   # E,F,G co-move through a shared wave (a confounder)
    )
    field = generate_planted_field(truth, S=30, T=60, noise=0.25, seed=1)

    # This prong tests the CPW S/L IDENTIFIABILITY (separability of a shared factor from
    # direct edges), which §III.4.1-2 defines without the §III.4(5) smoothness prior. The
    # prior is a distinct capability (borrow-strength shrinkage) with its own recovery test;
    # here it is turned off (γ=0) so the incoherence/mutual-exclusion mechanics are what is
    # scored, on clean planted data where no borrowing is needed.
    run = run_ldo(field, K=8, n_subsamples=10, stability_threshold=0.5, run_residual_scan=False,
                  gamma_temporal=0.0, gamma_disease=0.0, seed=0)
    score = recovery_score(run.link_records, truth)

    # every planted edge is recovered...
    assert score.recall == 1.0, score.details
    assert score.tp == 2
    # ...the shared factor is attributed as latent_shared, NOT a web of direct edges...
    assert score.factor_attribution == 1.0
    # ...precision stays high (the incoherence gate + S/L mutual exclusion keep the factor
    # out of the direct-edge set, so there is no separability leak)...
    assert score.precision >= 0.6
    # ...and where a lag was recovered it is within ±1 of the planted lag 6.
    assert score.lag_mae is None or score.lag_mae <= 1.5
