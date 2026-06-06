from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES, OutputSchemaRegistry, OutputValidationResult


def _read(path: Path):
    return pq.read_table(path)


def validate_output_bundle(
    *,
    run_dir: str,
    schema_registry: OutputSchemaRegistry | None = None,
) -> OutputValidationResult:
    schema_registry = schema_registry or OutputSchemaRegistry()
    root = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        return OutputValidationResult(ok=False, errors=[f"run_dir does not exist: {root}"], warnings=[])

    expected_names = {OUTPUT_BUNDLE_FILES[key] for key in schema_registry.required_keys}
    found_names = {p.name for p in root.iterdir()}

    missing = expected_names - found_names
    extra = found_names - expected_names

    for name in sorted(missing):
        errors.append(f"missing first-class artifact: {name}")
    for name in sorted(extra):
        errors.append(f"extra first-class artifact: {name}")

    if errors:
        return OutputValidationResult(ok=False, errors=errors, warnings=warnings)

    try:
        v = _read(root / "V_fields.parquet")
        q = _read(root / "Q_tensor.parquet")
        vd = _read(root / "VariableDictionary.parquet")
        edges = _read(root / "E_DAG.parquet")
        warnings_table = _read(root / "Warnings.parquet")
    except Exception as exc:
        return OutputValidationResult(ok=False, errors=[f"parquet read failure: {exc}"], warnings=warnings)

    if q.num_rows == 0:
        errors.append("Q_tensor is empty")

    v_ids = set(v.column("field_id").to_pylist()) if "field_id" in v.column_names else set()
    vd_ids = set(vd.column("field_id").to_pylist()) if "field_id" in vd.column_names else set()

    if not v_ids.issubset(vd_ids):
        errors.append("VariableDictionary does not cover all V_fields")

    if edges.num_rows:
        for col in ["parent_field_id", "child_field_id"]:
            if col in edges.column_names:
                bad = set(edges.column(col).to_pylist()) - v_ids
                if bad:
                    errors.append(f"E_DAG {col} contains IDs absent from V_fields: {sorted(bad)}")

    if warnings_table.num_rows and "field_id" in warnings_table.column_names:
        bad_warnings = {
            x for x in warnings_table.column("field_id").to_pylist()
            if x is not None and x not in v_ids
        }
        if bad_warnings:
            errors.append(f"Warnings link to invalid field IDs: {sorted(bad_warnings)}")

    try:
        manifest = json.loads((root / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
        telemetry = manifest["telemetry"]
        if telemetry.get("total_wall_seconds") is None or telemetry["total_wall_seconds"] < 0:
            errors.append("telemetry.total_wall_seconds missing or negative")
        for stage, status in telemetry.get("stage_status", {}).items():
            if status not in {"success", "skipped", "blocked", "failed"}:
                errors.append(f"invalid telemetry stage status: {stage}={status}")
        for stage, duration in telemetry.get("stage_wall_seconds", {}).items():
            if duration is None or duration < 0:
                errors.append(f"invalid telemetry stage duration: {stage}={duration}")
    except Exception as exc:
        errors.append(f"invalid ReproducibilityManifest.json telemetry: {exc}")

    return OutputValidationResult(ok=not errors, errors=errors, warnings=warnings)
