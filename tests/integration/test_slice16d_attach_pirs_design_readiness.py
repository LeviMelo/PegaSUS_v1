
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.pirs.design_readiness import attach_pirs_design_readiness_to_run


def _write_table(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_slice16d_attach_blocks_metadata_only_promoted_efg_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tables = run_dir / "Tables"
    tables.mkdir(parents=True)
    for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")

    field_id = "efg__sim_do__underlying_icd_norm"
    _write_table(
        run_dir / "V_fields.parquet",
        [
            {
                "field_id": field_id,
                "name": "SIM-DO underlying ICD metadata field",
                "state": "quarantined_descriptive",
                "materialization_state": "metadata_only",
                "dashboard_safe": "false",
            }
        ],
        pa.schema([
            ("field_id", pa.string()),
            ("name", pa.string()),
            ("state", pa.string()),
            ("materialization_state", pa.string()),
            ("dashboard_safe", pa.string()),
        ]),
    )
    _write_table(
        run_dir / "Q_tensor.parquet",
        [
            {
                "field_id": field_id,
                "state": "quarantined_descriptive",
                "warnings": json.dumps(["q_tensor_placeholder_no_numerical_tensor"]),
                "n_eff": 0.0,
                "missingness": 1.0,
            }
        ],
        pa.schema([
            ("field_id", pa.string()),
            ("state", pa.string()),
            ("warnings", pa.string()),
            ("n_eff", pa.float64()),
            ("missingness", pa.float64()),
        ]),
    )
    design_plan = {
        "schema_version": "1.0",
        "selected_outcome_field_id": field_id,
        "selected_covariate_field_ids": [],
        "selected_offset_field_id": None,
    }
    (tables / "pirs_design_plan.json").write_text(json.dumps(design_plan), encoding="utf-8")

    result = attach_pirs_design_readiness_to_run(run_dir=run_dir)
    assert result["pirs_design_readiness_gate"]["status"] == "blocked"
    manifest = json.loads((tables / "pirs_design_readiness.json").read_text(encoding="utf-8"))
    assert manifest["blocked_field_count"] == 1
    assert manifest["decisions"][0]["ready"] is False
    assert "materialization_state_not_tensor_backed:metadata_only" in manifest["decisions"][0]["reasons"]
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["pirs_design_readiness_gate"]["status"] == "blocked"


def test_slice16d_attach_accepts_materialized_verified_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tables = run_dir / "Tables"
    tables.mkdir(parents=True)
    for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")

    field_id = "sim__mortality_rate"
    _write_table(
        run_dir / "V_fields.parquet",
        [{"field_id": field_id, "state": "verified", "materialization_state": "materialized", "dashboard_safe": "true"}],
        pa.schema([("field_id", pa.string()), ("state", pa.string()), ("materialization_state", pa.string()), ("dashboard_safe", pa.string())]),
    )
    _write_table(
        run_dir / "Q_tensor.parquet",
        [{"field_id": field_id, "state": "verified", "warnings": json.dumps([]), "n_eff": 50.0, "missingness": 0.0}],
        pa.schema([("field_id", pa.string()), ("state", pa.string()), ("warnings", pa.string()), ("n_eff", pa.float64()), ("missingness", pa.float64())]),
    )
    (tables / "pirs_design_plan.json").write_text(
        json.dumps({"selected_outcome_field_id": field_id, "selected_covariate_field_ids": []}),
        encoding="utf-8",
    )

    result = attach_pirs_design_readiness_to_run(run_dir=run_dir)
    assert result["pirs_design_readiness_gate"]["status"] == "ready"
    manifest = json.loads((tables / "pirs_design_readiness.json").read_text(encoding="utf-8"))
    assert manifest["ready_field_count"] == 1
    assert manifest["blocked_field_count"] == 0
