"""Canonical output-table I/O over the PegaSUS storage boundary.

This module is intentionally small.  It gives production output attachers a
single row-oriented interface while keeping the parquet implementation inside
``pegasus.storage``.  Legacy fixture bundle writers may keep their historical
helpers until they are deleted or moved to a fixture namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa

from pegasus.storage import append_replace as storage_append_replace
from pegasus.storage import read_table as storage_read_table
from pegasus.storage import row_count as storage_row_count
from pegasus.storage import schema as storage_schema
from pegasus.storage import write_table as storage_write_table


def _rows(value: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(row) for row in (value or [])]


def _table_to_rows(table: pa.Table) -> list[dict[str, Any]]:
    return table.to_pylist()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a parquet table into row dictionaries through storage.read_table."""

    return _table_to_rows(storage_read_table(Path(path)))


def table_schema(path: str | Path) -> pa.Schema:
    """Return the persisted Arrow schema for an output artifact."""

    return storage_schema(Path(path))


def table_row_count(path: str | Path) -> int:
    """Return row count through the storage boundary."""

    return storage_row_count(Path(path))


def write_rows_like(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows, preserving an existing table schema when the file exists."""

    target = Path(path)
    materialized = _rows(rows)
    if target.exists():
        return storage_write_table(target, materialized, schema_policy="schema", schema=storage_schema(target))
    return storage_write_table(target, materialized, schema_policy="infer")


def empty_like(path: str | Path) -> Path:
    """Replace an existing table by an empty table with the same schema."""

    target = Path(path)
    return storage_write_table(target, [], schema_policy="schema", schema=storage_schema(target))


def append_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Append rows to an existing table while preserving its schema."""

    target = Path(path)
    incoming = _rows(rows)
    if not incoming:
        return target
    if not target.exists():
        return storage_write_table(target, incoming, schema_policy="infer")
    current = read_rows(target)
    return storage_write_table(target, current + incoming, schema_policy="schema", schema=storage_schema(target))


def _remove_existing(
    current: list[dict[str, Any]],
    *,
    id_column: str | None,
    incoming: list[dict[str, Any]],
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
    remove_column: str | None = None,
    remove_values: set[str] | None = None,
) -> list[dict[str, Any]]:
    incoming_ids = {str(row.get(id_column)) for row in incoming if id_column and row.get(id_column) is not None}
    explicit_ids = {str(value) for value in (remove_ids or set())}
    remove_values = {str(value) for value in (remove_values or set())}

    filtered: list[dict[str, Any]] = []
    for row in current:
        if id_column:
            value = str(row.get(id_column) or "")
            if value in incoming_ids or value in explicit_ids:
                continue
            if remove_prefixes and any(value.startswith(prefix) for prefix in remove_prefixes):
                continue
        if remove_column and remove_values and str(row.get(remove_column) or "") in remove_values:
            continue
        filtered.append(row)
    return filtered


def append_replace_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    id_column: str | None = None,
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
    remove_column: str | None = None,
    remove_values: set[str] | None = None,
) -> Path:
    """Append rows after removing existing rows selected by ID/prefix/value.

    This is a row-oriented adapter for run-bundle mutation.  It preserves an
    existing output schema when possible and delegates physical writes to
    ``pegasus.storage``.
    """

    target = Path(path)
    incoming = _rows(rows)
    if not target.exists():
        return storage_write_table(target, incoming, schema_policy="infer")
    current = read_rows(target)
    merged = _remove_existing(
        current,
        id_column=id_column,
        incoming=incoming,
        remove_ids=remove_ids,
        remove_prefixes=remove_prefixes,
        remove_column=remove_column,
        remove_values=remove_values,
    ) + incoming
    return storage_write_table(target, merged, schema_policy="schema", schema=storage_schema(target))


__all__ = [
    "append_replace_rows",
    "append_rows",
    "empty_like",
    "read_rows",
    "table_row_count",
    "table_schema",
    "write_rows_like",
]
