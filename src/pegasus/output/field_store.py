from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


class FieldStoreError(Exception):
    pass


def read_v_fields(run_dir: str | Path) -> pl.DataFrame:
    path = Path(run_dir) / "V_fields.parquet"

    if not path.exists():
        raise FieldStoreError(f"Missing V_fields table: {path}")

    return pl.read_parquet(path)


def list_fields(run_dir: str | Path) -> pl.DataFrame:
    v_fields = read_v_fields(run_dir)

    if v_fields.is_empty():
        return v_fields

    columns = [
        "field_id",
        "name",
        "kind",
        "carrier",
        "unit",
        "aggregation",
        "state",
        "dashboard_safe",
        "path",
    ]

    return v_fields.select([column for column in columns if column in v_fields.columns])


def get_field_row(run_dir: str | Path, field_id: str) -> dict[str, Any]:
    v_fields = read_v_fields(run_dir)
    hits = v_fields.filter(pl.col("field_id") == field_id)

    if hits.height == 0:
        raise FieldStoreError(f"Field not found in V_fields: {field_id}")

    if hits.height > 1:
        raise FieldStoreError(f"Duplicate field_id in V_fields: {field_id}")

    return hits.row(0, named=True)


def read_field_data(run_dir: str | Path, field_id: str) -> pl.DataFrame:
    row = get_field_row(run_dir, field_id)
    path_value = row.get("path")

    if path_value is None or path_value == "":
        raise FieldStoreError(f"Field has no materialized data path: {field_id}")

    data_path = Path(run_dir) / str(path_value)

    if not data_path.exists():
        raise FieldStoreError(f"Field data path does not exist: {data_path}")

    return pl.read_parquet(data_path)


def field_row_json(row: dict[str, Any], column: str) -> dict[str, Any]:
    value = row.get(column)

    if value is None or value == "":
        return {}

    if not isinstance(value, str):
        raise FieldStoreError(f"Expected JSON string in column {column}, got {type(value)}.")

    return json.loads(value)