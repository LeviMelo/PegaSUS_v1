from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from pegasus.storage import (
    append_replace,
    hash_table,
    read_table,
    row_count,
    scan_table,
    schema,
    write_table,
)


def test_storage_write_read_schema_and_hash_stability(tmp_path) -> None:
    path = tmp_path / "table.parquet"
    declared = pa.schema([("id", pa.string()), ("value", pa.int64())])
    rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
    write_table(path, rows, schema=declared)
    first_hash = hash_table(path)
    write_table(path, rows, schema_policy="preserve")
    assert hash_table(path) == first_hash
    assert schema(path).equals(declared)
    assert row_count(path) == 2
    assert read_table(path).to_pylist() == rows


def test_storage_append_replace_is_idempotent(tmp_path) -> None:
    path = tmp_path / "table.parquet"
    write_table(path, [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
    append_replace(path, [{"id": "b", "value": 3}], id_column="id")
    append_replace(path, [{"id": "b", "value": 3}], id_column="id")
    assert read_table(path).to_pylist() == [{"id": "a", "value": 1}, {"id": "b", "value": 3}]


def test_storage_scan_supports_projection_and_filter(tmp_path) -> None:
    path = tmp_path / "table.parquet"
    write_table(path, [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
    scanner = scan_table(path, columns=["id"], filters=pc.field("value") > 1)
    assert scanner.to_table().to_pylist() == [{"id": "b"}]


def test_storage_preserve_rejects_unconvertible_schema(tmp_path) -> None:
    path = tmp_path / "table.parquet"
    write_table(path, [{"id": "a", "value": 1}])
    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)):
        write_table(path, [{"id": "b", "value": "not-an-int"}], schema_policy="preserve")
