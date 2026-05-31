from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_json
from pegasus.output.schemas import PARQUET_FILENAMES, PARQUET_SCHEMAS, Schema
from pegasus.problem1.compiled import CompiledField
from pegasus.problem1.contracts import FieldNode, QState, WarningRecord


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _coerce(df: pl.DataFrame, schema: Schema) -> pl.DataFrame:
    out = df

    for name, dtype in schema.items():
        if name not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(name))

    return out.select([pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()])


def _upsert_parquet(
    path: Path,
    new_rows: pl.DataFrame,
    *,
    schema: Schema,
    key_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = _coerce(new_rows, schema)

    if path.exists():
        old = _coerce(pl.read_parquet(path), schema)
        combined = pl.concat([old, new_rows], how="vertical")
    else:
        combined = new_rows

    if key_columns:
        combined = combined.unique(subset=key_columns, keep="last", maintain_order=True)

    combined.write_parquet(path)


def _append_parquet(
    path: Path,
    new_rows: pl.DataFrame,
    *,
    schema: Schema,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = _coerce(new_rows, schema)

    if path.exists():
        old = _coerce(pl.read_parquet(path), schema)
        combined = pl.concat([old, new_rows], how="vertical")
    else:
        combined = new_rows

    combined.write_parquet(path)


def field_node_to_row(field: FieldNode) -> dict[str, Any]:
    lineage_hash = field.lineage.stable_hash()
    registry_hash = sha256_json(field.lineage.registry_versions)

    return {
        "field_id": field.id,
        "name": field.name,
        "kind": field.kind,
        "carrier": field.carrier,
        "unit": field.unit,
        "aggregation": field.aggregation,
        "role": field.role,
        "source": field.source,
        "support_json": _json(field.support),
        "axes_json": _json(field.axes),
        "operator": field.operator,
        "provenance": field.provenance,
        "state": field.state,
        "dashboard_safe": str(field.dashboard_safe),
        "warnings": field.warnings,
        "lineage_hash": lineage_hash,
        "registry_hash": registry_hash,
        "materialization_state": field.materialization_state,
        "path": field.path,
    }


def edge_rows_for_field(field: FieldNode) -> list[dict[str, Any]]:
    created_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for parent_id in field.lineage.parent_ids:
        payload = {
            "parent_field_id": parent_id,
            "child_field_id": field.id,
            "operator": field.lineage.operator_type,
            "operator_params": field.lineage.operator_params,
            "registry_versions": field.lineage.registry_versions,
        }

        rows.append(
            {
                "edge_id": sha256_json(payload),
                "parent_field_id": parent_id,
                "child_field_id": field.id,
                "operator": field.lineage.operator_type,
                "operator_params_json": _json(field.lineage.operator_params),
                "registry_versions_json": _json(field.lineage.registry_versions),
                "created_at": created_at,
            }
        )

    return rows


def q_state_to_row(q_state: QState) -> dict[str, Any]:
    return {
        "field_id": q_state.field_id,
        "n_events": q_state.n_events,
        "n_denom": q_state.n_denom,
        "n_eff": q_state.n_eff,
        "cov_S": q_state.cov_S,
        "cov_T": q_state.cov_T,
        "missingness": q_state.missingness,
        "zero_inflation": q_state.zero_inflation,
        "denom_fragility": q_state.denom_fragility,
        "cv": q_state.cv,
        "moran_i": q_state.moran_i,
        "temporal_roughness": q_state.temporal_roughness,
        "spatial_entropy": q_state.spatial_entropy,
        "provenance_risk": q_state.provenance_risk,
        "race_bridge_cv": q_state.race_bridge_cv,
        "sensitivity_width": q_state.sensitivity_width,
        "state": q_state.state,
        "dashboard_safe": str(q_state.dashboard_safe),
        "warnings": q_state.warnings,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "q_schema_version": "1.0",
    }


def warning_to_row(warning: WarningRecord) -> dict[str, Any]:
    return {
        "warning_id": warning.warning_id,
        "field_id": warning.field_id,
        "source": warning.source,
        "severity": warning.severity,
        "code": warning.code,
        "message": warning.message,
        "parent_warning_ids": warning.parent_warning_ids,
        "created_at": warning.created_at,
    }


def write_compiled_field(run_dir: str | Path, compiled: CompiledField) -> FieldNode:
    run_dir = Path(run_dir)

    field_data_dir = run_dir / "Tables" / "field_data"
    field_data_dir.mkdir(parents=True, exist_ok=True)

    relative_data_path = Path("Tables") / "field_data" / f"{compiled.field.id}.parquet"
    absolute_data_path = run_dir / relative_data_path
    compiled.data.write_parquet(absolute_data_path)

    field = compiled.field.model_copy(
        update={
            "path": str(relative_data_path),
            "materialization_state": "materialized",
        }
    )

    v_fields_row = pl.DataFrame([field_node_to_row(field)])
    q_row = pl.DataFrame([q_state_to_row(compiled.q_state)])

    _upsert_parquet(
        run_dir / PARQUET_FILENAMES["V_fields"],
        v_fields_row,
        schema=PARQUET_SCHEMAS["V_fields"],
        key_columns=["field_id"],
    )

    edge_rows = edge_rows_for_field(field)
    if edge_rows:
        _upsert_parquet(
            run_dir / PARQUET_FILENAMES["E_DAG"],
            pl.DataFrame(edge_rows),
            schema=PARQUET_SCHEMAS["E_DAG"],
            key_columns=["edge_id"],
        )

    _upsert_parquet(
        run_dir / PARQUET_FILENAMES["Q_tensor"],
        q_row,
        schema=PARQUET_SCHEMAS["Q_tensor"],
        key_columns=["field_id"],
    )

    if compiled.warnings:
        warning_rows = pl.DataFrame([warning_to_row(warning) for warning in compiled.warnings])
        _append_parquet(
            run_dir / PARQUET_FILENAMES["Warnings"],
            warning_rows,
            schema=PARQUET_SCHEMAS["Warnings"],
        )

    return field