from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        raise RuntimeError(f"Required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def backup(path: str, suffix: str = ".slice28z_storage_boundary.bak") -> None:
    p = ROOT / path
    b = p.with_name(p.name + suffix)
    if p.exists() and not b.exists():
        b.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")


def replace_top_level_function(text: str, name: str, replacement: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {name}("):
            start = i
            break
    if start is None:
        raise RuntimeError(f"top-level function not found: {name}")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line and not line.startswith((" ", "\t")) and (line.startswith("def ") or line.startswith("class ") or line.startswith("@")):
            end = j
            break
    new_lines = replacement.rstrip().splitlines()
    return "\n".join(lines[:start] + new_lines + lines[end:]) + "\n"


def ensure_import(text: str, import_block: str, *, after: str = "from typing import Any") -> str:
    if import_block in text:
        return text
    if after in text:
        return text.replace(after, after + "\n" + import_block, 1)
    # fallback after pathlib import
    marker = "from pathlib import Path"
    if marker in text:
        return text.replace(marker, marker + "\n" + import_block, 1)
    raise RuntimeError(f"cannot insert import block: {import_block}")


def remove_line_containing(text: str, token: str) -> str:
    return "\n".join(line for line in text.splitlines() if token not in line) + "\n"


TABLE_IO = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa

from pegasus.storage import append_replace, read_table, row_count, schema, write_table


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
    helper usable for first-write unit fixtures and auxiliary artifacts while
    retaining fixed-schema behavior for existing first-class bundle tables.
    """
    path = Path(path)
    payload = _rows(rows)
    if not path.exists():
        return write_rows(path, payload)
    table_schema_obj = schema(path)
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
    """Append/replace rows by id through the storage boundary.

    ``pegasus.storage.append_replace`` is authoritative for existing tables.  For
    first writes, infer the table schema from the incoming rows so callers do not
    need to create an empty seed table only to replace into it.
    """
    path = Path(path)
    payload = _rows(rows)
    if not path.exists():
        return write_rows(path, payload)
    append_replace(path, payload, id_column=id_column)
    return path


def empty_like(path: str | Path) -> Path:
    """Overwrite an existing table with zero rows while preserving its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cannot create empty_like for missing table: {path}")
    return write_rows_like(path, [])
'''


AUDIT = r'''
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()

PRODUCTION_STORAGE_CLEAN = (
    Path("src/pegasus/efg/promotion_apply.py"),
    Path("src/pegasus/output/maternal_child_compile_attach.py"),
    Path("src/pegasus/output/population_tensor_compile_attach.py"),
    Path("src/pegasus/output/race_bridge_attach.py"),
    Path("src/pegasus/dashboard/read_only.py"),
)

FORBIDDEN = (
    "import pyarrow.parquet as pq",
    "pq.read_table",
    "pq.write_table",
    "pl.read_parquet",
    ".write_parquet(",
)


def main() -> None:
    errors: list[str] = []
    findings: list[str] = []
    for rel in PRODUCTION_STORAGE_CLEAN:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"missing expected file: {rel}")
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for token in FORBIDDEN:
                if token in line:
                    findings.append(f"{rel}:{line_no}: {line.strip()}")
    if findings:
        errors.extend(f"storage boundary bypass remains: {item}" for item in findings)
    payload = {
        "audit": "slice28z_storage_boundary_adoption",
        "status": "failed" if errors else "passed",
        "errors": errors,
        "checked_files": [str(path) for path in PRODUCTION_STORAGE_CLEAN],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


TEST_TABLE_IO = r'''
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
'''


TEST_AUDIT = r'''
from __future__ import annotations

import json
import subprocess
import sys


def test_slice28z_storage_adoption_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28z_storage_boundary_adoption.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
'''


def patch_promotion_apply() -> None:
    path = "src/pegasus/efg/promotion_apply.py"
    backup(path)
    text = read(path)
    text = ensure_import(text, "from pegasus.output.table_io import append_replace_rows, read_rows, write_rows_like")
    text = replace_top_level_function(text, "_read_rows", dedent('''
        def _read_rows(path: Path) -> list[dict[str, Any]]:
            return read_rows(path)
    '''))
    text = replace_top_level_function(text, "_write_rows", dedent('''
        def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
            write_rows_like(path, rows)
    '''))
    text = replace_top_level_function(text, "_append_replace", dedent('''
        def _append_replace(path: Path, new_rows: list[dict[str, Any]], *, id_column: str) -> None:
            append_replace_rows(path, new_rows, id_column=id_column)
    '''))
    text = remove_line_containing(text, "import pyarrow.parquet as pq")
    write(path, text)


def patch_population_tensor_compile_attach() -> None:
    path = "src/pegasus/output/population_tensor_compile_attach.py"
    backup(path)
    text = read(path)
    text = ensure_import(text, "from pegasus.output.table_io import append_replace_rows, read_rows, write_rows, write_rows_like")
    text = replace_top_level_function(text, "_read_rows", dedent('''
        def _read_rows(path: Path) -> list[dict[str, Any]]:
            return read_rows(path)
    '''))
    text = replace_top_level_function(text, "_write_rows_like", dedent('''
        def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
            write_rows_like(path, rows)
    '''))
    text = replace_top_level_function(text, "_append_replace", dedent('''
        def _append_replace(
            path: Path,
            rows: list[dict[str, Any]],
            *,
            id_column: str,
            remove_ids: set[str] | None = None,
            remove_prefixes: tuple[str, ...] = (),
        ) -> None:
            existing = read_rows(path)
            if remove_ids or remove_prefixes:
                filtered: list[dict[str, Any]] = []
                for row in existing:
                    value = str(row.get(id_column, ""))
                    if remove_ids and value in remove_ids:
                        continue
                    if remove_prefixes and any(value.startswith(prefix) for prefix in remove_prefixes):
                        continue
                    filtered.append(row)
                write_rows_like(path, filtered)
            append_replace_rows(path, rows, id_column=id_column)
    '''))
    text = text.replace('pl.DataFrame([meta]).write_parquet(run_dir / "Tables" / "population_tensor_diagnostics.parquet")',
                        'write_rows(run_dir / "Tables" / "population_tensor_diagnostics.parquet", [meta])')
    text = remove_line_containing(text, "import pyarrow.parquet as pq")
    write(path, text)


def patch_maternal_child_compile_attach() -> None:
    path = "src/pegasus/output/maternal_child_compile_attach.py"
    backup(path)
    text = read(path)
    text = ensure_import(text, "from pegasus.output.table_io import append_rows, read_rows, write_rows, write_rows_like")
    text = replace_top_level_function(text, "_read_rows", dedent('''
        def _read_rows(path: Path) -> list[dict[str, Any]]:
            return read_rows(path)
    '''))
    text = replace_top_level_function(text, "_write_rows_like", dedent('''
        def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
            write_rows_like(path, rows)
    '''))
    text = replace_top_level_function(text, "_append_rows", dedent('''
        def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str, remove_values: set[str]) -> None:
            existing = [row for row in read_rows(path) if str(row.get(remove_column)) not in remove_values]
            write_rows_like(path, existing + list(rows))
    '''))
    text = text.replace('pl.DataFrame([row]).write_parquet(table_path)', 'write_rows(table_path, [row])')
    text = remove_line_containing(text, "import pyarrow.parquet as pq")
    write(path, text)


def patch_race_bridge_attach() -> None:
    path = "src/pegasus/output/race_bridge_attach.py"
    backup(path)
    text = read(path)
    text = ensure_import(text, "from pegasus.output.table_io import read_rows, write_rows, write_rows_like")
    text = replace_top_level_function(text, "_append_rows", dedent('''
        def _append_rows(path: Path, rows: list[dict[str, Any]], *, id_column: str | None = None, remove_values: set[str] | None = None) -> None:
            existing = read_rows(path)
            if id_column and remove_values:
                existing = [row for row in existing if str(row.get(id_column)) not in remove_values]
            if id_column:
                incoming = {str(row.get(id_column)): row for row in rows}
                merged: list[dict[str, Any]] = []
                seen: set[str] = set()
                for row in existing:
                    key = str(row.get(id_column))
                    if key in incoming:
                        merged.append(incoming[key])
                        seen.add(key)
                    else:
                        merged.append(row)
                for key, row in incoming.items():
                    if key not in seen:
                        merged.append(row)
                write_rows_like(path, merged)
            else:
                write_rows_like(path, existing + list(rows))
    '''))
    text = text.replace('pl.DataFrame(summary_rows).write_parquet(summary_path)', 'write_rows(summary_path, summary_rows)')
    write(path, text)


def patch_dashboard_read_only() -> None:
    path = "src/pegasus/dashboard/read_only.py"
    backup(path)
    text = read(path)
    text = ensure_import(text, "from pegasus.output.table_io import read_rows, table_row_count, table_schema")
    text = replace_top_level_function(text, "parquet_row_count", dedent('''
        def parquet_row_count(path: Path) -> int:
            return table_row_count(path)
    '''))
    text = replace_top_level_function(text, "read_table_head", dedent('''
        def read_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
            assert_no_compute_trigger("table_head")
            path = _table_path(_run_dir(run_dir), table_name)
            rows = read_rows(path)[:limit]
            return {"table": table_name, "path": str(path), "limit": limit, "rows": rows}
    '''))
    text = remove_line_containing(text, "import pyarrow.parquet as pq")
    write(path, text)


def main() -> None:
    write("src/pegasus/output/table_io.py", TABLE_IO)
    patch_promotion_apply()
    patch_population_tensor_compile_attach()
    patch_maternal_child_compile_attach()
    patch_race_bridge_attach()
    patch_dashboard_read_only()
    write("scripts/dev/audits/audit_slice28z_storage_boundary_adoption.py", AUDIT)
    write("tests/unit/test_slice28z_output_table_io.py", TEST_TABLE_IO)
    write("tests/integration/test_slice28z_storage_boundary_audit.py", TEST_AUDIT)
    print("Applied Slice 28Z storage boundary adoption.")
    print("Changed production modules:")
    for item in (
        "src/pegasus/output/table_io.py",
        "src/pegasus/efg/promotion_apply.py",
        "src/pegasus/output/maternal_child_compile_attach.py",
        "src/pegasus/output/population_tensor_compile_attach.py",
        "src/pegasus/output/race_bridge_attach.py",
        "src/pegasus/dashboard/read_only.py",
        "scripts/dev/audits/audit_slice28z_storage_boundary_adoption.py",
        "tests/unit/test_slice28z_output_table_io.py",
        "tests/integration/test_slice28z_storage_boundary_audit.py",
    ):
        print(f"  {item}")


if __name__ == "__main__":
    main()

# slice28z repair v3: dashboard/read_only.py restored table_head contract via AST patcher.
