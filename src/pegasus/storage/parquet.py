"""Canonical Arrow/Parquet storage operations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SchemaPolicy = Literal["infer", "preserve", "strict"]


def _table(value: Any, *, schema: pa.Schema | None = None) -> pa.Table:
    if isinstance(value, pa.Table):
        table = value
    elif isinstance(value, pa.RecordBatch):
        table = pa.Table.from_batches([value])
    else:
        table = pa.Table.from_pylist(list(value), schema=schema)
    if schema is not None and not table.schema.equals(schema, check_metadata=False):
        table = table.cast(schema)
    return table


def read_table(
    path: str | Path,
    columns: list[str] | tuple[str, ...] | None = None,
    filters: Any = None,
) -> pa.Table:
    return pq.read_table(Path(path), columns=columns, filters=filters)


def scan_table(
    path: str | Path,
    columns: list[str] | tuple[str, ...] | None = None,
    filters: Any = None,
) -> ds.Scanner:
    dataset = ds.dataset(Path(path), format="parquet")
    return dataset.scanner(columns=columns, filter=filters)


def write_table(
    path: str | Path,
    rows_or_table: Any,
    *,
    schema_policy: SchemaPolicy = "infer",
    schema: pa.Schema | None = None,
    compression: str = "zstd",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_schema = pq.read_schema(target) if target.exists() else None
    effective_schema = schema
    if schema_policy in {"preserve", "strict"} and existing_schema is not None:
        effective_schema = existing_schema
    table = _table(rows_or_table, schema=effective_schema if schema_policy != "infer" else schema)
    if schema_policy == "strict" and effective_schema is not None and not table.schema.equals(effective_schema, check_metadata=False):
        raise ValueError(f"strict schema mismatch for {target}")
    pq.write_table(table, target, compression=compression)
    return target


def row_count(path: str | Path) -> int:
    target = Path(path)
    if not target.exists():
        return 0
    return int(pq.ParquetFile(target).metadata.num_rows)


def schema(path: str | Path) -> pa.Schema:
    return pq.read_schema(Path(path))


def hash_table(path: str | Path) -> str:
    table = read_table(path)
    digest = hashlib.sha256()
    digest.update(str(table.schema.remove_metadata()).encode("utf-8"))
    for batch in table.to_batches(max_chunksize=8192):
        digest.update(json.dumps(batch.to_pylist(), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return digest.hexdigest()


def append_replace(
    path: str | Path,
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    id_column: str,
) -> Path:
    """Append rows to a parquet table, replacing existing rows with matching ids."""
    target = Path(path)
    incoming = [dict(row) for row in rows]
    if not incoming:
        return target
    if not target.exists():
        return write_table(target, incoming)
    existing = read_table(target).to_pylist()
    incoming_ids = {row.get(id_column) for row in incoming}
    kept = [row for row in existing if row.get(id_column) not in incoming_ids]
    return write_table(target, kept + incoming, schema_policy="preserve")
