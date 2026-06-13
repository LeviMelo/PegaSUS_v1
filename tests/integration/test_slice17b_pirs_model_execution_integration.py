
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from pegasus.cli import app
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.pirs_execute import run_execute_pirs_model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _ready_run(run_dir: Path) -> None:
    create_empty_output_bundle(run_dir)
    matrix = run_dir / "Tables" / "pirs_design_matrix.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"row_id": 0, "intercept": 1.0, "response": 10.0, "covariate_001": 1.0},
        {"row_id": 1, "intercept": 1.0, "response": 12.0, "covariate_001": 2.0},
        {"row_id": 2, "intercept": 1.0, "response": 14.0, "covariate_001": 3.0},
        {"row_id": 3, "intercept": 1.0, "response": 16.0, "covariate_001": 4.0},
    ]), matrix)
    _write_json(run_dir / "Tables" / "pirs_design_matrix_manifest.json", {"status": "ready", "matrix_written": True, "matrix_path": str(matrix), "family": "gaussian_identity", "residual_mode": "in_sample", "fold_scheme": "none_in_sample_fast_budget", "field_specs": [{"field_id": "deaths", "role": "outcome", "column": "response"}, {"field_id": "capacity", "role": "covariate", "column": "covariate_001"}]})


def test_slice17b_executes_model_mutates_bundle_and_validates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _ready_run(run_dir)
    result = run_execute_pirs_model(run_dir=run_dir)
    gate = result["pirs_model_execution_gate"]
    assert gate["status"] == "fitted"
    assert gate["mutated_output_bundle"] is True
    assert gate["residual_field_id"]
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors
    v_rows = pq.read_table(run_dir / "V_fields.parquet").to_pylist()
    assert any(row["field_id"] == gate["residual_field_id"] and row["kind"] == "model_residual" for row in v_rows)
    model_rows = pq.read_table(run_dir / "ModelAssociations.parquet").to_pylist()
    residual_rows = pq.read_table(run_dir / "ResidualAssociations.parquet").to_pylist()
    assert any(row.get("id") == gate["model_id"] for row in model_rows)
    assert any(row.get("id") == gate["residual_field_id"] for row in residual_rows)
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert manifest["pirs_model_execution_gate"]["status"] == "fitted"


def test_slice17b_cli_executes_and_inspects_model_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cli"
    _ready_run(run_dir)
    runner = CliRunner()
    result = runner.invoke(app, ["pirs", "execute-model", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert '"status": "fitted"' in result.output
    manifest = run_dir / "Tables" / "pirs_model_execution_manifest.json"
    assert manifest.exists()
    result = runner.invoke(app, ["pirs", "inspect-model-execution", "--manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    assert '"model_fit_state": "fitted"' in result.output
