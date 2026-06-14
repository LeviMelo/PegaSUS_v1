"""Optional DuckDB query adapter."""

from __future__ import annotations

from typing import Any

from pegasus.core.exceptions import StorageBackendError


def query_arrow(sql: str, *, parameters: list[Any] | None = None):
    try:
        import duckdb
    except ImportError as exc:
        raise StorageBackendError("DuckDB backend is unavailable") from exc
    connection = duckdb.connect(database=":memory:")
    try:
        relation = connection.execute(sql, parameters or [])
        return relation.fetch_arrow_table()
    finally:
        connection.close()
