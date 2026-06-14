from __future__ import annotations

from pegasus.output.sidra_denominator_anchor import _append_rows, _get_field_by_name, _remove_by_values
from pegasus.output.table_io import read_rows, write_rows


def test_slice28za_row_helpers_preserve_append_replace_semantics(tmp_path):
    path = tmp_path / "table.parquet"
    write_rows(
        path,
        [
            {"id": "a", "name": "old", "value": 1},
            {"id": "b", "name": "keep", "value": 2},
        ],
    )

    _append_rows(path, [{"id": "a", "name": "new", "value": 3}], remove_column="id", remove_values={"a"})

    rows = read_rows(path)
    assert [row["id"] for row in rows] == ["b", "a"]
    assert _get_field_by_name(rows, "new")["value"] == 3
    assert _remove_by_values(rows, "id", {"b"}) == [{"id": "a", "name": "new", "value": 3}]


def test_slice28za_sidra_anchor_source_uses_storage_boundary():
    import pegasus.output.sidra_denominator_anchor as module

    text = module.__loader__.get_source(module.__name__)
    assert "pl.read_parquet" not in text
    assert ".write_parquet" not in text
    assert "pyarrow.parquet" not in text
    assert "from pegasus.output.table_io import" in text
