
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
from pegasus.workflows.hsic_execute import run_execute_hsic_residual_scan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _ready_modeled_run(run_dir: Path) -> None:
    create_empty_output_bundle(run_dir)
    matrix = run_dir / "Tables" / "pirs_design_matrix.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"row_id": 0, "intercept": 1.0, "response": 10.0, "covariate_001": 1.0, "covariate_002": 4.0},
        {"row_id": 1, "intercept": 1.0, "response": 12.0, "covariate_001": 2.0, "covariate_002": 3.0},
        {"row_id": 2, "intercept": 1.0, "response": 13.0, "covariate_001": 3.0, "covariate_002": 2.0},
        {"row_id": 3, "intercept": 1.0, "response": 17.0, "covariate_001": 4.0, "covariate_002": 1.0},
    ]), matrix)
    _write_json(run_dir / "Tables" / "pirs_design_matrix_manifest.json", {
        "status": "ready",
        "matrix_written": True,
        "matrix_path": str(matrix),
        "family": "gaussian_identity",
        "residual_mode": "in_sample",
        "fold_scheme": "none_in_sample_fast_budget",
        "field_specs": [
            {"field_id": "deaths", "role": "outcome", "column": "response"},
            {"field_id": "capacity_a", "role": "covariate", "column": "covariate_001"},
            {"field_id": "capacity_b", "role": "covariate", "column": "covariate_002"},
        ],
    })
    run_execute_pirs_model(run_dir=run_dir)


def test_slice18a_scans_residuals_writes_hypotheses_and_validates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _ready_modeled_run(run_dir)
    result = run_execute_hsic_residual_scan(run_dir=run_dir)
    gate = result["hsic_residual_scan_gate"]
    assert gate["status"] == "scanned"
    assert gate["hypothesis_count"] == 2
    assert (run_dir / "Tables" / "hsic_residual_scan_scores.parquet").exists()
    assert (run_dir / "Tables" / "hsic_residual_scan_manifest.json").exists()
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors
    hypotheses = pq.read_table(run_dir / "Hypotheses.parquet").to_pylist()
    assert len(hypotheses) == 2
    assert {row["hsic_mode"] for row in hypotheses} == {"exact_linear"}
    assert {row["covariate_field_id"] for row in hypotheses} == {"capacity_a", "capacity_b"}
    assert {row["residual_field_id"] for row in hypotheses} == {gate["residual_field_id"]}
    assert {row["fdr_method"] for row in hypotheses} == {"BH"}
    assert all(row["n_eff"] == 4.0 for row in hypotheses)
    assert all(row["approximation_diagnostics_json"] for row in hypotheses)
    repro = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert repro["hsic_residual_scan_gate"]["status"] == "scanned"
    assert repro["telemetry"]["stage_status"]["pirs_hsic"] == "success"


def test_slice18a_cli_scans_and_inspects(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_cli"
    _ready_modeled_run(run_dir)
    runner = CliRunner()
    result = runner.invoke(app, ["pirs", "scan-residuals-hsic", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert '"status": "scanned"' in result.output
    manifest = run_dir / "Tables" / "hsic_residual_scan_manifest.json"
    result = runner.invoke(app, ["pirs", "inspect-hsic-scan", "--manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    assert '"hypothesis_count": 2' in result.output
