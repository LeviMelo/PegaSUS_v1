
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.pirs.model_execution import build_pirs_model_execution_manifest, inspect_pirs_model_execution_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def test_slice17b_blocks_when_design_matrix_manifest_is_not_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "Tables").mkdir(parents=True)
    _write_json(run_dir / "Tables" / "pirs_design_matrix_manifest.json", {"status": "blocked", "matrix_written": False})
    payload = build_pirs_model_execution_manifest(run_dir=run_dir, mutate_output_bundle=False)
    assert payload["status"] == "blocked"
    assert payload["model_fit_state"] == "blocked"
    assert payload["mutated_output_bundle"] is False
    assert any(str(reason).startswith("design_matrix_manifest_not_ready") for reason in payload["blocking_reasons"])


def test_slice17b_fits_design_matrix_and_writes_model_artifacts_without_bundle_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "Tables").mkdir(parents=True)
    matrix = run_dir / "Tables" / "pirs_design_matrix.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"row_id": 0, "intercept": 1.0, "response": 1.0, "covariate_001": 1.0},
        {"row_id": 1, "intercept": 1.0, "response": 2.0, "covariate_001": 2.0},
        {"row_id": 2, "intercept": 1.0, "response": 3.0, "covariate_001": 3.0},
    ]), matrix)
    _write_json(run_dir / "Tables" / "pirs_design_matrix_manifest.json", {"status": "ready", "matrix_written": True, "matrix_path": str(matrix), "family": "gaussian_identity", "field_specs": [{"field_id": "field:y", "role": "outcome", "column": "response"}, {"field_id": "field:x", "role": "covariate", "column": "covariate_001"}]})
    payload = build_pirs_model_execution_manifest(run_dir=run_dir, mutate_output_bundle=False)
    assert payload["status"] == "fitted"
    assert payload["model_fit_state"] == "fitted"
    assert payload["residual_state"] == "materialized"
    assert payload["coefficient_count"] == 2
    assert Path(payload["coefficients_path"]).exists()
    assert Path(payload["residual_values_path"]).exists()
    inspected = inspect_pirs_model_execution_manifest(run_dir / "Tables" / "pirs_model_execution_manifest.json")
    assert inspected["status"] == "fitted"
