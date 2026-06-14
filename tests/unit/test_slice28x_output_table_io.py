from __future__ import annotations

from pegasus.output.table_io import append_replace_rows, empty_like, read_rows, table_row_count, write_rows_like


def test_slice28x_table_io_write_read_append_replace_and_empty(tmp_path):
    path = tmp_path / "table.parquet"
    write_rows_like(path, [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
    assert read_rows(path) == [{"id": "a", "value": 1}, {"id": "b", "value": 2}]

    append_replace_rows(path, [{"id": "b", "value": 20}, {"id": "c", "value": 3}], id_column="id")
    rows = read_rows(path)
    assert rows == [{"id": "a", "value": 1}, {"id": "b", "value": 20}, {"id": "c", "value": 3}]
    assert table_row_count(path) == 3

    empty_like(path)
    assert read_rows(path) == []
    assert table_row_count(path) == 0
