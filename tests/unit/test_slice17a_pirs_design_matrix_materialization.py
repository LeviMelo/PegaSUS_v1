
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.pirs.design_matrix import (
    build_pirs_design_matrix_manifest,
    inspect_pirs_design_matrix_manifest,
    pirs_design_matrix_summary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _write_q(run_dir: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), run_dir / "Q_tensor.parquet")


def test_slice17a_blocks_when_design_readiness_is_not_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "Tables").mkdir(parents=True)
    _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "planned", "outcome_field_id": "field:y", "covariate_field_ids": ["field:x"]})
    _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "blocked", "ready": False})
    _write_q(run_dir, [{"field_id": "field:y", "values_json": "[1, 2]"}, {"field_id": "field:x", "values_json": "[3, 4]"}])

    payload = build_pirs_design_matrix_manifest(run_dir=run_dir)

    assert payload["status"] == "blocked"
    assert payload["matrix_written"] is False
    assert any(str(reason).startswith("design_readiness_not_ready") for reason in payload["blocking_reasons"])


def test_slice17a_materializes_numeric_design_matrix_when_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "Tables").mkdir(parents=True)
    _write_json(
        run_dir / "Tables" / "pirs_design_plan.json",
        {
            "status": "planned",
            "family": "poisson_rate",
            "residual_mode": "cross_fitted",
            "outcome_field_id": "field:y",
            "covariate_field_ids": ["field:x1", "field:x2"],
            "offset_field_id": "field:pop",
        },
    )
    _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "ready", "ready": True})
    _write_q(
        run_dir,
        [
            {"field_id": "field:y", "values_json": "[10, 20, 30]"},
            {"field_id": "field:x1", "values_json": "[1, 2, 3]"},
            {"field_id": "field:x2", "values_json": "[4, 5, 6]"},
            {"field_id": "field:pop", "values_json": "[100, 200, 300]"},
        ],
    )

    payload = build_pirs_design_matrix_manifest(run_dir=run_dir)

    assert payload["status"] == "ready"
    assert payload["design_matrix_state"] == "materialized"
    assert payload["row_count"] == 3
    assert payload["columns"] == ["row_id", "intercept", "response", "covariate_001", "covariate_002", "offset"]
    rows = pq.read_table(run_dir / "Tables" / "pirs_design_matrix.parquet").to_pylist()
    assert rows[0]["response"] == 10.0
    assert rows[0]["intercept"] == 1.0
    summary = pirs_design_matrix_summary(payload)
    assert summary["model_fit_state"] == "not_started"
    inspected = inspect_pirs_design_matrix_manifest(run_dir / "Tables" / "pirs_design_matrix_manifest.json")
    assert inspected["status"] == "ready"
