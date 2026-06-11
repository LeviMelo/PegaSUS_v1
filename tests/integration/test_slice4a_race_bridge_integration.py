from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile
from pegasus.workflows.race_bridge import run_attach_race_bridge


def test_slice4a_attach_race_bridge_to_compile_run(tmp_path: Path):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    compile_result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
        data_root=data_root,
    )
    assert compile_result["validation"].ok, compile_result["validation"].errors

    sim_events = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
    result = run_attach_race_bridge(
        run_dir=run_dir,
        sim_events_path=sim_events,
        bridge_prior_path="tests/fixtures/race_bridge/fixedC_valid.json",
        municipality_cod6="270430",
    )
    assert result["validation"].ok, result["validation"].errors
    assert validate_output_bundle(run_dir=str(run_dir)).ok

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    field_ids = set(v["field_id"].to_list())
    assert "SIMRaceAdminRawCount_4" in field_ids
    assert "SIMRaceBridgeMissingRaceObserver" in field_ids
    assert "SIMRaceBridgePosteriorCount_parda" in field_ids
    assert "SIMRaceBridgePosteriorCount_branca" in field_ids

    posterior = v.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").to_dicts()[0]
    assert "Bridge_R_fixedC_dynamic_weight" in posterior["operator"]
    assert "missing_race_share" in posterior["axes_json"]
    assert "bayesian_ecological_bridge_warning" in posterior["axes_json"]

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    parda_q = q.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").to_dicts()[0]
    assert parda_q["state"] in {"fragile", "verified"}

    table = pl.read_parquet(run_dir / "Tables" / "race_bridge_summary.parquet")
    assert table.height == 5
    assert set(table["race_target_category"].to_list()) == {"branca", "preta", "amarela", "parda", "indigena"}
