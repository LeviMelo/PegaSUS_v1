
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pegasus.output.table_io import append_replace_rows, read_rows, write_rows, write_rows_like

import pyarrow as pa
import polars as pl

from pegasus.core.hashing import sha256_file
from pegasus.output.population_tensor_bundle import (
    _failed_dense_branch,
    _field,
    _q_row,
    _vd_row,
    _warning_rows,
)
from pegasus.she.population.solvers import solve_population_tensor_from_sidra_anchor

POPULATION_TENSOR_FIELD_PREFIX = "population_tensor_"
POPULATION_TENSOR_WARNING_PREFIX = "population_tensor_"
POPULATION_TENSOR_FAILED_BRANCH_IDS = {"failed_dense_national_population_tensor_above_threshold"}



def _read_rows(path: Path) -> list[dict[str, Any]]:
    return read_rows(path)

def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows_like(path, rows)

def _append_replace(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    id_column: str,
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
) -> None:
    existing = read_rows(path)
    if remove_ids or remove_prefixes:
        filtered: list[dict[str, Any]] = []
        for row in existing:
            value = str(row.get(id_column, ""))
            if remove_ids and value in remove_ids:
                continue
            if remove_prefixes and any(value.startswith(prefix) for prefix in remove_prefixes):
                continue
            filtered.append(row)
        write_rows_like(path, filtered)
    append_replace_rows(path, rows, id_column=id_column)
def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _metadata(*, result, field: dict[str, Any], sidra_facts_path: Path, mode: str) -> dict[str, Any]:
    manifest = result.as_manifest()
    manifest.update(
        {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "population_solver",
            "field_id": field["field_id"],
            "field_name": field["name"],
            "requested_population_mode": mode,
            "sidra_facts_path": str(sidra_facts_path),
            "source_hashes": {"sidra_facts": sha256_file(sidra_facts_path)},
            "independent_denominator_mode": result.mode == "independent_denominator",
            "sim_feedback_warning": bool(result.denominator_feedback_warning),
            "dashboard_safe": field.get("dashboard_safe"),
            "materialization_state": field.get("materialization_state"),
            "table_paths": {"diagnostics": "Tables/population_tensor_diagnostics.parquet"},
        }
    )
    return manifest


def attach_population_tensor_compile_fields(
    *,
    run_dir: str | Path,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sidra_facts_path = Path(sidra_facts_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not sidra_facts_path.exists():
        raise FileNotFoundError(f"sidra_facts_path does not exist: {sidra_facts_path}")

    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=sidra_facts_path, mode=mode)
    field = _field(result)
    q_row = _q_row(field, result)
    vd_row = _vd_row(field, result)
    warning_rows = _warning_rows(field, result)
    failed_row = _failed_dense_branch(field)
    meta = _metadata(result=result, field=field, sidra_facts_path=sidra_facts_path, mode=mode)

    _append_replace(run_dir / "V_fields.parquet", [field], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "Q_tensor.parquet", [q_row], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "VariableDictionary.parquet", [vd_row], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "Warnings.parquet", warning_rows, id_column="warning_id", remove_prefixes=(POPULATION_TENSOR_WARNING_PREFIX,))
    _append_replace(run_dir / "FailedBranches.parquet", [failed_row], id_column="failed_branch_id", remove_ids=POPULATION_TENSOR_FAILED_BRANCH_IDS)

    (run_dir / "Tables").mkdir(exist_ok=True)
    write_rows(run_dir / "Tables" / "population_tensor_diagnostics.parquet", [meta])

    for json_name in ["RunConfig.json", "ReproducibilityManifest.json", "P_vector.json"]:
        path = run_dir / json_name
        payload = _load_json(path)
        payload["population_tensor"] = meta
        if json_name == "P_vector.json":
            payload.setdefault("source_systems", [])
            if "SIDRA" not in payload["source_systems"]:
                payload["source_systems"].append("SIDRA")
        _write_json(path, payload)

    return meta
