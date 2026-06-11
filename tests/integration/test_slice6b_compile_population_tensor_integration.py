from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


def _field_ids(run_dir: Path) -> set[str]:
    return set(pl.read_parquet(run_dir / "V_fields.parquet")["field_id"].to_list())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_slice6b_compile_attaches_independent_population_tensor(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_population_tensor"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_population_tensor.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors

    ids = _field_ids(run_dir)
    assert "population_tensor_independent_denominator" in ids

    run_config = _load_json(run_dir / "RunConfig.json")
    manifest = _load_json(run_dir / "ReproducibilityManifest.json")
    assert run_config["population_tensor"]["mode"] == "independent_denominator"
    assert run_config["population_tensor"]["attach_stage"] == "population_solver"
    assert manifest["population_tensor"]["mode"] == "independent_denominator"
    assert manifest["telemetry"]["stage_status"]["population_solver"] == "success"
    assert (run_dir / "Tables" / "population_tensor_diagnostics.parquet").exists()

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    assert "population_tensor_independent_denominator" in set(q["field_id"].to_list())

    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors


def test_slice6b_compile_baseline_remains_official_anchor_without_population_tensor(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_baseline"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors
    assert not any(field_id.startswith("population_tensor_") for field_id in _field_ids(run_dir))

    manifest = _load_json(run_dir / "ReproducibilityManifest.json")
    assert manifest["telemetry"]["stage_status"]["population_solver"] == "blocked"


def test_slice6b_compile_sim_informed_population_tensor_is_warning_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_sim_informed_population_tensor"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_population_tensor_sim_informed.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors

    ids = _field_ids(run_dir)
    assert "population_tensor_sim_informed_denominator" in ids

    fields = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
    field = next(row for row in fields if row["field_id"] == "population_tensor_sim_informed_denominator")
    assert field["dashboard_safe"] != "true"
    assert field["state"] in {"fragile", "experimental", "blocked", "quarantined"}

    warnings = pl.read_parquet(run_dir / "Warnings.parquet").to_dicts()
    assert any(row.get("code") == "sim_informed_population_feedback_risk" for row in warnings)

    run_config = _load_json(run_dir / "RunConfig.json")
    assert run_config["population_tensor"]["sim_feedback_warning"] is True
    assert validate_output_bundle(run_dir=str(run_dir)).ok


def test_slice6b_validator_rejects_population_tensor_metadata_erasure(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_population_tensor_invalid"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke_population_tensor.json",
        run_dir=run_dir,
    )
    assert result["validation"].ok, result["validation"].errors

    run_config_path = run_dir / "RunConfig.json"
    run_config = _load_json(run_config_path)
    del run_config["population_tensor"]["solver_id"]
    run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    validation = validate_output_bundle(run_dir=str(run_dir))
    assert not validation.ok
    assert any("RunConfig.population_tensor missing keys" in error for error in validation.errors)
