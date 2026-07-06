"""LDO Rung-1 causal orientation wired into run_ldo (§IV, CAUSAL-01).

pegasus.causal.orient (LiNGAM non-Gaussian orientation) was built and unit-tested but
never called by the LDO — every emitted edge was undirected. run_ldo now orients
contemporaneous edges on the RAW (non-gaussianized) values: where a variable is
non-Gaussian, LiNGAM identifies the causal direction (source ← cause); where both are
Gaussian it honestly reports undirected. These pin the wire.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.orchestrator import run_ldo


def _field(a: np.ndarray, b: np.ndarray, noise: np.ndarray, S: int, T: int) -> LDOField:
    X = np.stack([a, b, noise])
    return LDOField(
        variables=("A", "B", "N"),
        space_ids=tuple(str(500000 + i) for i in range(S)),  # codes absent from adjacency → no whitening
        time_ids=tuple(range(T)),
        X=X, W=np.ones_like(X), resolution="year",
    )


def _ab_record(run):
    return next((r for r in run.link_records if {r.source_var, r.target_var} == {"A", "B"}
                and r.edge_type == "contemporaneous"), None)


def test_non_gaussian_edge_is_oriented_toward_the_cause():
    rng = np.random.default_rng(0)
    S, T = 30, 50
    a = rng.uniform(-2.0, 2.0, (S, T))          # strongly non-Gaussian cause
    b = 0.9 * a + 0.25 * rng.standard_normal((S, T))   # A → B
    run = run_ldo(_field(a, b, rng.standard_normal((S, T)), S, T), K=1, run_residual_scan=False, seed=0)
    assert run.diagnostics["n_oriented_lingam"] >= 1
    rec = _ab_record(run)
    assert rec is not None, "A-B contemporaneous edge should be recovered"
    assert "oriented_non_gaussian_lingam" in rec.warnings
    assert rec.source_var == "A" and rec.target_var == "B", \
        f"LiNGAM should orient A→B (cause=A); got {rec.source_var}→{rec.target_var}"


def test_gaussian_edge_is_left_undirected():
    rng = np.random.default_rng(1)
    S, T = 30, 50
    a = rng.standard_normal((S, T))             # Gaussian
    b = 0.9 * a + 0.25 * rng.standard_normal((S, T))   # Gaussian → direction unidentifiable
    run = run_ldo(_field(a, b, rng.standard_normal((S, T)), S, T), K=1, run_residual_scan=False, seed=0)
    rec = _ab_record(run)
    assert rec is not None
    assert "orientation_undirected_unidentifiable" in rec.warnings, \
        "a Gaussian edge must be honestly reported undirected, not falsely oriented"
