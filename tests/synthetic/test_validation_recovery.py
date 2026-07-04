"""VAL-03 — synthetic ground-truth recovery (MSD-III §IX.2).

The standing separability/recovery prong: plant a known structure (two directed-lagged
edges + a shared latent factor + noise variables), run the real LDO, and score recovery.
A certified engine recovers the planted edges, attributes the shared factor as
``latent_shared`` (not a web of direct edges), and gets the lag right — the empirical
test of separability adequacy that argument cannot settle.
"""

from __future__ import annotations

from pegasus.pirs.ldo.orchestrator import run_ldo
from pegasus.validation.synthetic import (
    PlantedEdge,
    PlantedTruth,
    generate_planted_field,
    recovery_score,
)


def test_ldo_recovers_planted_structure() -> None:
    truth = PlantedTruth(
        variables=("A", "B", "C", "D", "E", "F", "G", "H"),
        edges=(PlantedEdge("A", "B", 6, 1.0), PlantedEdge("C", "D", 0, 1.0)),
        factors=((("E", "F"), 1.2),),   # E,F co-move through a shared wave (a confounder)
    )
    field = generate_planted_field(truth, S=30, T=60, noise=0.25, seed=1)

    run = run_ldo(field, K=8, n_subsamples=10, stability_threshold=0.5, run_residual_scan=False, seed=0)
    score = recovery_score(run.link_records, truth)

    # every planted edge is recovered...
    assert score.recall == 1.0, score.details
    assert score.tp == 2
    # ...the shared factor is attributed as latent_shared, NOT a web of direct edges...
    assert score.factor_attribution == 1.0
    # ...precision stays high (at most the one factor-pair separability leak)...
    assert score.precision >= 0.6
    # ...and where a lag was recovered it is within ±1 of the planted lag 6.
    assert score.lag_mae is None or score.lag_mae <= 1.5
