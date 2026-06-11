import json
import shutil
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile
from pegasus.workflows.race_bridge import run_plan_race_bridge


def test_baseline_compile_remains_decoupled_and_valid(tmp_path: Path):
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=tmp_path / "baseline_run",
        data_root=tmp_path / "baseline_data",
    )
    assert result["validation"].ok, result["validation"].errors
    assert result["race_bridge"] is None
    v = pl.read_parquet(tmp_path / "baseline_run" / "V_fields.parquet")
    assert "SIMRaceBridgePosteriorCount_parda" not in set(v["field_id"].to_list())
    manifest = json.loads((tmp_path / "baseline_run" / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert manifest["telemetry"]["stage_status"]["race_bridge"] == "skipped"


def test_registry_backed_compile_attaches_downstream_bridge(tmp_path: Path):
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_race_bridge.json",
        run_dir=tmp_path / "bridge_run",
        data_root=tmp_path / "bridge_data",
    )
    assert result["status"] == "success"
    assert result["validation"].ok, result["validation"].errors
    assert result["race_bridge"] is not None
    assert result["race_bridge_plan"]["status"] == "planned"
    v = pl.read_parquet(tmp_path / "bridge_run" / "V_fields.parquet")
    field_ids = set(v["field_id"].to_list())
    assert "SIMRaceAdminRawCount_4" in field_ids
    assert "SIMRaceBridgeMissingRaceObserver" in field_ids
    assert "SIMRaceBridgePosteriorCount_parda" in field_ids
    q = pl.read_parquet(tmp_path / "bridge_run" / "Q_tensor.parquet")
    parda_q = q.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").to_dicts()[0]
    assert parda_q["dashboard_safe"] == "False"
    assert parda_q["state"] == "fragile"
    if "bridge_mode" in q.columns:
        assert parda_q["bridge_mode"] == "fixedC_dynamic_weight"
    manifest = json.loads((tmp_path / "bridge_run" / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    run_config = json.loads((tmp_path / "bridge_run" / "RunConfig.json").read_text(encoding="utf-8"))
    assert manifest["race_bridge"]["raw_admin_counts_preserved"] is True
    assert run_config["race_bridge"]["missing_category_preserved"] is True
    assert "race_bridge_prior" in manifest["source_hashes"]
    assert "race_bridge_summary" in manifest["source_hashes"]
    assert manifest["telemetry"]["stage_status"]["race_bridge"] == "success"


def test_registry_backed_plan_cli_workflow_payload_after_compile(tmp_path: Path):
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=tmp_path / "run",
        data_root=tmp_path / "data",
    )
    assert result["validation"].ok, result["validation"].errors
    sim_events = tmp_path / "data" / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
    plan = run_plan_race_bridge(
        sim_events_path=sim_events,
        intent_path="config/intents/alagoas_smoke_race_bridge.json",
        registry_path="config/registries/race_bridge_priors.yaml",
        municipality_cod6="270430",
    )
    assert plan["summary"]["bridge_id"] == "fixedC_sim_admin_to_ibge_selfdeclared_smoke_v1"
    assert "posterior_counts" in plan["summary"]
    assert plan["summary"]["plan"]["status"] == "planned"


def test_validator_rejects_race_bridge_fields_without_metadata(tmp_path: Path):
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_race_bridge.json",
        run_dir=tmp_path / "run",
        data_root=tmp_path / "data",
    )
    assert result["validation"].ok, result["validation"].errors
    poisoned = tmp_path / "poisoned"
    shutil.copytree(tmp_path / "run", poisoned)
    run_config_path = poisoned / "RunConfig.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config.pop("race_bridge", None)
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    validation = validate_output_bundle(run_dir=str(poisoned))
    assert not validation.ok
    assert any("race_bridge metadata" in error or "RunConfig.race_bridge" in error for error in validation.errors)
