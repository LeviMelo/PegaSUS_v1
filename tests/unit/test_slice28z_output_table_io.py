
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.table_io import append_replace_rows, empty_like, read_rows, table_row_count, write_rows_like, write_rows


def test_slice28z_write_rows_like_preserves_existing_schema(tmp_path):
    path = tmp_path / "table.parquet"
    schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.int64()), pa.field("extra", pa.string())])
    pq.write_table(pa.Table.from_pylist([{"id": "a", "value": 1, "extra": "x"}], schema=schema), path)

    write_rows_like(path, [{"id": "b", "value": 2, "ignored": "drop"}])

    out = pq.read_table(path)
    assert out.schema.names == ["id", "value", "extra"]
    assert out.to_pylist() == [{"id": "b", "value": 2, "extra": None}]


def test_slice28z_append_replace_rows_dedupes_by_id(tmp_path):
    path = tmp_path / "table.parquet"
    schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.int64())])
    pq.write_table(pa.Table.from_pylist([{"id": "a", "value": 1}, {"id": "b", "value": 2}], schema=schema), path)

    append_replace_rows(path, [{"id": "b", "value": 20}, {"id": "c", "value": 3}], id_column="id")

    assert read_rows(path) == [{"id": "a", "value": 1}, {"id": "b", "value": 20}, {"id": "c", "value": 3}]


def test_slice28z_empty_like_and_row_count(tmp_path):
    path = tmp_path / "table.parquet"
    schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.int64())])
    pq.write_table(pa.Table.from_pylist([{"id": "a", "value": 1}], schema=schema), path)
    empty_like(path)
    assert table_row_count(path) == 0
    assert pq.read_table(path).schema.names == ["id", "value"]


def test_slice28z_write_rows_infers_new_table(tmp_path):
    path = tmp_path / "aux.parquet"
    write_rows(path, [{"stage": "x", "count": 1}])
    assert read_rows(path) == [{"stage": "x", "count": 1}]
