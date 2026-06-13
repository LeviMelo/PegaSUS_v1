
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _write_table(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def main() -> int:
    errors: list[str] = []
    from pegasus.pirs.design_readiness import attach_pirs_design_readiness_to_run, design_readiness_rejection_reasons
    from pegasus.workflows.pirs_readiness import run_attach_pirs_design_readiness_to_run

    reasons = design_readiness_rejection_reasons(
        field={"field_id": "x", "state": "quarantined_descriptive", "materialization_state": "metadata_only", "dashboard_safe": False},
        q_state={"field_id": "x", "state": "quarantined_descriptive", "warnings": ["q_tensor_placeholder_no_numerical_tensor"], "n_eff": 0},
    )
    if "materialization_state_not_tensor_backed:metadata_only" not in reasons:
        errors.append("metadata_only fields are not rejected")
    if "q_tensor_placeholder:q_tensor_placeholder_no_numerical_tensor" not in reasons:
        errors.append("placeholder Q tensor warning is not rejected")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        tables = run_dir / "Tables"
        tables.mkdir(parents=True)
        for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
            (run_dir / name).write_text("{}", encoding="utf-8")
        field_id = "efg__audit"
        _write_table(
            run_dir / "V_fields.parquet",
            [{"field_id": field_id, "state": "quarantined_descriptive", "materialization_state": "metadata_only", "dashboard_safe": "false"}],
            pa.schema([("field_id", pa.string()), ("state", pa.string()), ("materialization_state", pa.string()), ("dashboard_safe", pa.string())]),
        )
        _write_table(
            run_dir / "Q_tensor.parquet",
            [{"field_id": field_id, "state": "quarantined_descriptive", "warnings": json.dumps(["q_tensor_placeholder_no_numerical_tensor"]), "n_eff": 0.0}],
            pa.schema([("field_id", pa.string()), ("state", pa.string()), ("warnings", pa.string()), ("n_eff", pa.float64())]),
        )
        (tables / "pirs_design_plan.json").write_text(json.dumps({"selected_outcome_field_id": field_id}), encoding="utf-8")
        result = attach_pirs_design_readiness_to_run(run_dir=run_dir)
        workflow_result = run_attach_pirs_design_readiness_to_run(run_dir=run_dir)
        if result["pirs_design_readiness_gate"]["status"] != "blocked":
            errors.append("direct attach did not block metadata-only promoted EFG field")
        if workflow_result["pirs_design_readiness_gate"]["status"] != "blocked":
            errors.append("workflow attach did not block metadata-only promoted EFG field")
        if not (tables / "pirs_design_readiness.json").exists():
            errors.append("pirs_design_readiness.json was not written")

    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16D PIRS design-readiness gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
