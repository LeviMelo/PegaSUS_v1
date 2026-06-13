
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def test_slice17a_attach_blocks_empty_bundle_without_numerical_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "blocked", "warnings": ["empty_bundle"]})
    _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "blocked", "ready": False})

    result = run_attach_pirs_design_matrix_to_run(run_dir=run_dir)

    gate = result["pirs_design_matrix_gate"]
    assert gate["status"] == "blocked"
    assert gate["matrix_written"] is False
    assert (run_dir / "Tables" / "pirs_design_matrix_manifest.json").exists()
    assert not (run_dir / "Tables" / "pirs_design_matrix.parquet").exists()
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["pirs_design_matrix_gate"]["status"] == "blocked"


def test_slice17a_attach_writes_matrix_for_ready_synthetic_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "planned", "outcome_field_id": "y", "covariate_field_ids": ["x"]})
    _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "ready", "ready": True})
    pq.write_table(pa.Table.from_pylist([
        {"field_id": "y", "values_json": "[1, 0, 1]"},
        {"field_id": "x", "values_json": "[10, 20, 30]"},
    ]), run_dir / "Q_tensor.parquet")

    result = run_attach_pirs_design_matrix_to_run(run_dir=run_dir)

    gate = result["pirs_design_matrix_gate"]
    assert gate["status"] == "ready"
    assert gate["row_count"] == 3
    assert (run_dir / "Tables" / "pirs_design_matrix.parquet").exists()
    rows = pq.read_table(run_dir / "Tables" / "pirs_design_matrix.parquet").to_pylist()
    assert rows[2]["covariate_001"] == 30.0
