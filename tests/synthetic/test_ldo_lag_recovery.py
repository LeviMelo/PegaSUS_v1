"""LDO planted-lag recovery — the decisive proof-of-life (MSD-III §XI.4, TDD §5.8).

Generate a monthly synthetic field with a KNOWN lag-L directed edge A→B (plus
independent noise variables), run the *real* built LDO (`run_ldo`), and assert it
recovers the lag WITHOUT any variable being hand-named. This is the earliest test
that proves the engine does what the redesign exists for: discover a directed,
distributed-lag relationship from data alone.

If this is green, the LDO's lag-discovery capability is validated on ground truth.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.orchestrator import run_ldo


def _planted_lag_field(
    *, lag: int, coef: float = 0.8, noise: float = 0.25, n_noise: int = 2,
    S: int = 24, T: int = 48, seed: int = 1,
) -> LDOField:
    """A[s,t] iid; B[s,t] = coef·A[s,t-lag] + noise; plus independent noise vars."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((S, T))
    B = np.empty((S, T))
    B[:, :lag] = noise * rng.standard_normal((S, lag))
    B[:, lag:] = coef * A[:, : T - lag] + noise * rng.standard_normal((S, T - lag))
    layers = [A, B] + [rng.standard_normal((S, T)) for _ in range(n_noise)]
    X = np.stack(layers, axis=0)                       # (p, S, T)
    variables = ("A", "B") + tuple(f"noise{i}" for i in range(n_noise))
    return LDOField(
        variables=variables,
        space_ids=tuple(str(270000 + i) for i in range(S)),
        time_ids=tuple(range(T)),
        X=X,
        W=np.ones_like(X),
        resolution="month",
    )


def test_ldo_recovers_planted_lag() -> None:
    lag = 7
    field = _planted_lag_field(lag=lag)

    run = run_ldo(
        field, K=9, n_subsamples=8, stability_threshold=0.6,
        run_residual_scan=False, seed=0,
    )
    lagged = [r for r in run.link_records if r.edge_type == "lagged_directed"]

    # the planted directed edge A(t-lag) → B(t) is recovered...
    ab = [r for r in lagged if r.source_var == "A" and r.target_var == "B"]
    assert ab, f"A→B lagged edge not found; lagged edges: {[(r.source_var, r.target_var, r.lag_k) for r in lagged]}"
    best = max(ab, key=lambda r: abs(r.weight))
    assert abs(best.lag_k - lag) <= 1, f"recovered lag {best.lag_k}, expected ≈{lag}"
    assert abs(best.weight) > 0.2, f"planted edge too weak: {best.weight}"
    assert best.certification_status == "selected"

    # ...and no pure-noise pair is promoted to a selected directed edge
    for r in lagged:
        if r.certification_status == "selected":
            assert not (r.source_var.startswith("noise") and r.target_var.startswith("noise")), (
                f"spurious selected edge among noise variables: {r.source_var}→{r.target_var}"
            )


def test_ldo_direction_is_not_reversed() -> None:
    """The engine must not report B→A (the reverse of the planted direction) as the winner."""
    lag = 6
    field = _planted_lag_field(lag=lag, seed=3)
    run = run_ldo(field, K=9, n_subsamples=8, stability_threshold=0.6, run_residual_scan=False, seed=0)
    lagged = [r for r in run.link_records if r.edge_type == "lagged_directed" and r.certification_status == "selected"]
    ab = max((r for r in lagged if r.source_var == "A" and r.target_var == "B"), key=lambda r: abs(r.weight), default=None)
    ba = max((r for r in lagged if r.source_var == "B" and r.target_var == "A"), key=lambda r: abs(r.weight), default=None)
    assert ab is not None, "planted A→B edge not selected"
    # the correct direction must dominate any reverse artifact
    assert ba is None or abs(ab.weight) >= abs(ba.weight)
