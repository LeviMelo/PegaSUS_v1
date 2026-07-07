"""O8 / §IV — Rung-2 ITS promotions must pass a negative-control-outcomes bias check.

A genuine interruption effect is specific to the outcome. If the break at t0 ALSO fires on many
unrelated (negative-control) series, it is a common shock (a confounder), not the edge's effect —
so the Rung-2 promotion is VETOED (kept at Rung 1). An edge-specific break is promoted, recording
the negative-control clearance as an assumption.
"""

from __future__ import annotations

import numpy as np

from pegasus.causal.quasi import escalate_rung2_its, negative_control_break_fraction
from pegasus.ldo.records import LinkRecord


def _series(jump_at, jump, T=24, seed=0, base_slope=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = base_slope * t + 0.2 * rng.standard_normal(T)
    if jump_at is not None:
        y = y + (t >= jump_at) * jump
    return y


def test_negative_control_fraction_high_for_common_shock():
    # every series jumps at t=12 → a common shock
    sbv = {v: _series(12, 4.0, seed=i) for i, v in enumerate("ABCDE")}
    frac, n = negative_control_break_fraction(sbv, 12, exclude={"A", "B"})
    assert n >= 3 and frac >= 0.5           # controls C,D,E also break → common shock


def test_edge_specific_break_has_low_negative_control_fraction():
    sbv = {"A": _series(None, 0.0), "B": _series(12, 5.0)}  # only B jumps
    sbv.update({v: _series(None, 0.0, seed=i + 10) for i, v in enumerate("CDE")})  # flat controls
    frac, n = negative_control_break_fraction(sbv, 12, exclude={"A", "B"})
    assert n >= 3 and frac < 0.5


def _rung1_edge(src, tgt):
    return LinkRecord(source_var=src, target_var=tgt, edge_type="lagged_directed", lag_k=1,
                      weight=0.4, stability=0.9, uncertainty=0.05, causal_rung=1,
                      certification_status="selected")


def test_common_shock_break_is_vetoed_not_promoted():
    sbv = {v: _series(12, 4.0, seed=i) for i, v in enumerate("ABCDE")}
    out = escalate_rung2_its([_rung1_edge("A", "B")], sbv)
    r = out[0]
    assert r.causal_rung == 1                # NOT promoted to Rung 2
    assert any(w.startswith("rung2_vetoed_negative_control_common_shock") for w in r.warnings)


def test_edge_specific_break_is_promoted_to_rung2():
    sbv = {"A": _series(None, 0.0), "B": _series(12, 6.0)}
    sbv.update({v: _series(None, 0.0, seed=i + 10) for i, v in enumerate("CDE")})
    out = escalate_rung2_its([_rung1_edge("A", "B")], sbv)
    r = out[0]
    assert r.causal_rung == 2
    assert "negative_control_outcomes" in r.causal_assumptions
    assert any(w.startswith("rung2_negative_control_clear") for w in r.warnings)
