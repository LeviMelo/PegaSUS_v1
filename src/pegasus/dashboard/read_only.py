"""Read-only run-bundle inspection services for the PegaSUS dashboard.

These functions only read local files from completed run directories. They do
not fetch DATASUS/SIDRA, do not call compiler workflows, and do not mutate runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pegasus.output.table_io import read_rows, table_row_count, table_schema


from pegasus.dashboard.contracts import assert_no_compute_trigger
from pegasus.output.validate import validate_output_bundle


FIRST_CLASS_KEYS: tuple[str, ...] = (
    "V_fields",
    "E_DAG",
    "Q_tensor",
    "P_vector",
    "UserIntent",
    "Warnings",
    "ModelAssociations",
    "ResidualAssociations",
    "Hypotheses",
    "Tables",
    "Maps",
    "VariableDictionary",
    "FailedBranches",
    "QuarantinedFields",
    "ForcedFields",
    "RunConfig",
    "ReproducibilityManifest",
)

TABLE_FILES: dict[str, str] = {
    "V_fields": "V_fields.parquet",
    "E_DAG": "E_DAG.parquet",
    "Q_tensor": "Q_tensor.parquet",
    "Warnings": "Warnings.parquet",
    "ModelAssociations": "ModelAssociations.parquet",
    "ResidualAssociations": "ResidualAssociations.parquet",
    "Hypotheses": "Hypotheses.parquet",
    "VariableDictionary": "VariableDictionary.parquet",
    "FailedBranches": "FailedBranches.parquet",
    "QuarantinedFields": "QuarantinedFields.parquet",
    "ForcedFields": "ForcedFields.parquet",
}


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(path: str | Path) -> Path:
    run_dir = Path(path)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _table_path(run_dir: Path, table_name: str) -> Path:
    if table_name not in TABLE_FILES:
        raise ValueError(f"Unsupported dashboard table: {table_name!r}")
    path = run_dir / TABLE_FILES[table_name]
    if not path.exists():
        raise FileNotFoundError(f"Dashboard table missing: {path}")
    return path



def parquet_row_count(path: Path) -> int:
    return table_row_count(path)
def list_tables(*, run_dir: str | Path) -> dict[str, Any]:
    assert_no_compute_trigger("list_tables")
    root = _run_dir(run_dir)
    tables = []
    for key, rel in TABLE_FILES.items():
        path = root / rel
        tables.append({
            "name": key,
            "path": rel,
            "exists": path.exists(),
            "rows": parquet_row_count(path) if path.exists() else None,
        })
    extra_tables = []
    tables_dir = root / "Tables"
    if tables_dir.exists():
        for path in sorted(tables_dir.glob("*.parquet")):
            extra_tables.append({
                "name": path.stem,
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "rows": parquet_row_count(path),
            })
    return {"tables": tables, "artifact_tables": extra_tables}



def read_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
    assert_no_compute_trigger("table_head")
    if limit < 0 or limit > 500:
        raise ValueError("Dashboard table head limit must be between 0 and 500.")
    root = _run_dir(run_dir)
    path = _table_path(root, table_name)
    rows = read_rows(path)[:limit]
    schema_obj = table_schema(path)
    return {
        "table": table_name,
        "path": TABLE_FILES[table_name],
        "rows_returned": len(rows),
        "row_count": table_row_count(path),
        "columns": list(schema_obj.names),
        "rows": rows,
    }
def inspect_run(*, run_dir: str | Path, validate: bool = True) -> dict[str, Any]:
    assert_no_compute_trigger("inspect_run")
    root = _run_dir(run_dir)
    present = []
    missing = []
    for key in FIRST_CLASS_KEYS:
        if key == "Tables":
            exists = (root / "Tables").exists()
        elif key == "Maps":
            exists = (root / "Maps").exists()
        elif key in {"RunConfig", "UserIntent", "P_vector", "ReproducibilityManifest"}:
            filename = "ReproducibilityManifest.json" if key == "ReproducibilityManifest" else f"{key}.json"
            exists = (root / filename).exists()
        else:
            exists = (root / f"{key}.parquet").exists()
        (present if exists else missing).append(key)
    validation = validate_output_bundle(run_dir=str(root)) if validate else None
    run_config = _json_file(root / "RunConfig.json") if (root / "RunConfig.json").exists() else {}
    manifest = _json_file(root / "ReproducibilityManifest.json") if (root / "ReproducibilityManifest.json").exists() else {}
    table_summary = list_tables(run_dir=root)
    return {
        "run_dir": str(root),
        "read_only": True,
        "first_class_keys_present": present,
        "first_class_keys_missing": missing,
        "validation_ok": validation.ok if validation is not None else None,
        "validation_errors": validation.errors if validation is not None else [],
        "validation_warnings": validation.warnings if validation is not None else [],
        "run_config_slice": run_config.get("slice"),
        "workflow_mode": run_config.get("workflow_mode") or manifest.get("workflow_mode"),
        "telemetry_stage_status": (manifest.get("telemetry") or {}).get("stage_status", {}),
        "tables": table_summary,
    }


def variable_dictionary(*, run_dir: str | Path, limit: int = 100) -> dict[str, Any]:
    assert_no_compute_trigger("variable_dictionary")
    return read_table_head(run_dir=run_dir, table_name="VariableDictionary", limit=limit)


def hypotheses(*, run_dir: str | Path, limit: int = 100) -> dict[str, Any]:
    assert_no_compute_trigger("hypotheses")
    return read_table_head(run_dir=run_dir, table_name="Hypotheses", limit=limit)


def warnings_summary(*, run_dir: str | Path, limit: int = 100) -> dict[str, Any]:
    assert_no_compute_trigger("warnings")
    return read_table_head(run_dir=run_dir, table_name="Warnings", limit=limit)


def bundle_overview(*, run_dir: str | Path, limit: int = 20) -> dict[str, Any]:
    """Return a bundle-bound operational overview without triggering computation."""
    assert_no_compute_trigger("bundle_overview")
    if limit < 0 or limit > 500:
        raise ValueError("Dashboard overview limit must be between 0 and 500.")
    root = _run_dir(run_dir).resolve()
    inspected = inspect_run(run_dir=root, validate=True)
    run_config = _json_file(root / "RunConfig.json")
    manifest = _json_file(root / "ReproducibilityManifest.json")
    tables = root / "Tables"
    hsic_paths = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in tables.glob("*hsic*")
        if path.is_file()
    ) if tables.exists() else []
    return {
        "run_dir": str(root),
        "read_only": True,
        "validation_ok": inspected["validation_ok"],
        "source_reality": run_config.get("source_artifact_reality") or manifest.get("source_artifact_reality") or {},
        "registry_hashes": manifest.get("registry_hashes") or {},
        "efg": {
            "fields": read_table_head(run_dir=root, table_name="V_fields", limit=limit),
            "edges": read_table_head(run_dir=root, table_name="E_DAG", limit=limit),
        },
        "warnings": warnings_summary(run_dir=root, limit=limit),
        "hypotheses": hypotheses(run_dir=root, limit=limit),
        "hsic_artifacts": hsic_paths,
    }
