"""W-RACE-1 — the race-bridge posterior uncertainty (race_bridge_cv) propagates into the LDO
observation weight, so a race-uncertain field is down-weighted, not modelled as exact.
"""

from __future__ import annotations

import polars as pl

from pegasus.workflows.investigate import state_reliability_weights


def _q_tensor(run_dir, rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(run_dir / "Q_tensor.parquet")


def test_race_bridge_cv_downweights_reliability(tmp_path):
    run = tmp_path / "run"
    _q_tensor(run, [
        {"field_id": "plain", "n_events": 1000.0, "n_denom": None, "n_eff": 1000.0,
         "denom_fragility": 0.0, "provenance_risk": 0.0, "race_bridge_cv": None},
        {"field_id": "bridged", "n_events": 1000.0, "n_denom": None, "n_eff": 1000.0,
         "denom_fragility": 0.0, "provenance_risk": 0.0, "race_bridge_cv": 0.5},
    ])
    w = state_reliability_weights(run)
    assert w["plain"] == 1.0                          # no bridge, no deflation
    assert abs(w["bridged"] - 1.0 / 1.5) < 1e-9       # 1/(1+0.5) deflation
    assert w["bridged"] < w["plain"]                  # race-uncertain field down-weighted


def test_no_race_bridge_cv_is_a_noop(tmp_path):
    # every field lacks race_bridge_cv -> identical to the legacy weight (byte-identical no-op)
    run = tmp_path / "run"
    _q_tensor(run, [
        {"field_id": "a", "n_events": 500.0, "n_denom": None, "n_eff": 250.0,
         "denom_fragility": 0.2, "provenance_risk": 0.0, "race_bridge_cv": None},
    ])
    w = state_reliability_weights(run)
    assert abs(w["a"] - (250.0 / 500.0) * (1.0 - 0.2)) < 1e-9   # kish*(1-frag), no cv term
