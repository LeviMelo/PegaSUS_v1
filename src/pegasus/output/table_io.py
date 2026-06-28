from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa

from pegasus.storage import read_table, row_count, schema, write_table


def _rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _coerce_row_to_schema(row: dict[str, Any], table_schema: pa.Schema) -> dict[str, Any]:
    return {name: row.get(name) for name in table_schema.names}


def _table_from_rows(rows: list[dict[str, Any]], table_schema: pa.Schema | None = None) -> pa.Table:
    if table_schema is not None:
        payload = [_coerce_row_to_schema(row, table_schema) for row in rows]
        return pa.Table.from_pylist(payload, schema=table_schema)
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table({})


def _schema_with_promoted_nulls(rows: list[dict[str, Any]], table_schema: pa.Schema) -> pa.Schema:
    fields = []
    for field in table_schema:
        if not pa.types.is_null(field.type):
            fields.append(field)
            continue
        values = [row.get(field.name) for row in rows if row.get(field.name) is not None]
        if not values:
            fields.append(field)
            continue
        inferred = pa.array(values).type
        fields.append(pa.field(field.name, inferred, nullable=True, metadata=field.metadata))
    return pa.schema(fields, metadata=table_schema.metadata)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a Parquet table as row dictionaries through the storage boundary."""
    return read_table(path).to_pylist()


def table_schema(path: str | Path) -> pa.Schema:
    """Return table schema through the storage boundary."""
    return schema(path)


def table_row_count(path: str | Path) -> int:
    """Return row count through the storage boundary."""
    return row_count(path)


def write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows by inferring an Arrow schema.

    This is for newly-created auxiliary tables. Existing first-class bundle tables
    should normally use ``write_rows_like`` so their schema remains fixed.
    """
    path = Path(path)
    write_table(path, _table_from_rows(_rows(rows)))
    return path


def write_rows_like(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Overwrite a table, preserving its schema when the table already exists.

    When the table does not exist yet, fall back to ``write_rows``. This keeps the
    helper usable for first-write auxiliary artifacts while
    retaining fixed-schema behavior for existing first-class bundle tables.
    """
    path = Path(path)
    payload = _rows(rows)
    if not path.exists():
        return write_rows(path, payload)
    table_schema_obj = _schema_with_promoted_nulls(payload, schema(path))
    write_table(path, _table_from_rows(payload, table_schema_obj))
    return path


def append_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Append rows, preserving an existing schema when present."""
    path = Path(path)
    payload = _rows(rows)
    if not payload:
        if path.exists():
            return path
        return write_rows(path, [])
    if not path.exists():
        return write_rows(path, payload)
    existing = read_rows(path)
    return write_rows_like(path, existing + payload)


def append_replace_rows(path: str | Path, rows: Iterable[dict[str, Any]], *, id_column: str) -> Path:
    """Append rows while replacing existing rows with matching ids."""
    path = Path(path)
    payload = _rows(rows)
    if not payload:
        return path
    if not path.exists():
        return write_rows(path, payload)
    incoming_ids = {row.get(id_column) for row in payload}
    kept = [row for row in read_rows(path) if row.get(id_column) not in incoming_ids]
    return write_rows_like(path, kept + payload)


def empty_like(path: str | Path) -> Path:
    """Overwrite an existing table with zero rows while preserving its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cannot create empty_like for missing table: {path}")
    return write_rows_like(path, [])
