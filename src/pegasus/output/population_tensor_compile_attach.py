
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
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
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_table(path).schema
    shaped = [{name: row.get(name) for name in schema.names} for row in rows]
    if shaped:
        table = pa.Table.from_pylist(shaped, schema=schema)
    else:
        table = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    pq.write_table(table, path)


def _append_replace(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    id_column: str,
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
) -> None:
    remove_ids = remove_ids or set()
    existing = _read_rows(path)

    def keep(row: dict[str, Any]) -> bool:
        value = str(row.get(id_column, ""))
        if value in remove_ids:
            return False
        return not any(value.startswith(prefix) for prefix in remove_prefixes)

    _write_rows_like(path, [row for row in existing if keep(row)] + rows)


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
    pl.DataFrame([meta]).write_parquet(run_dir / "Tables" / "population_tensor_diagnostics.parquet")

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
