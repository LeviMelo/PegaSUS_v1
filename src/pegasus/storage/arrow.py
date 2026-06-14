"""Arrow conversion helpers used by the storage boundary."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


def as_arrow_table(value: Any, *, schema: pa.Schema | None = None) -> pa.Table:
    if isinstance(value, pa.Table):
        return value.cast(schema) if schema is not None else value
    if isinstance(value, pa.RecordBatch):
        table = pa.Table.from_batches([value])
        return table.cast(schema) if schema is not None else table
    return pa.Table.from_pylist(list(value), schema=schema)
