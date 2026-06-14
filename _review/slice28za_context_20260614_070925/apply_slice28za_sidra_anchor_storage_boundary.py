from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()
SIDRA_ANCHOR = ROOT / "src" / "pegasus" / "output" / "sidra_denominator_anchor.py"
AUDIT = ROOT / "scripts" / "dev" / "audits" / "audit_slice28za_sidra_anchor_storage_boundary.py"
TEST_UNIT = ROOT / "tests" / "unit" / "test_slice28za_sidra_anchor_storage_boundary.py"
TEST_INT = ROOT / "tests" / "integration" / "test_slice28za_sidra_anchor_storage_boundary_audit.py"
DOC = ROOT / "docs" / "production_boundaries.md"


def _backup(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _replace_top_level_function(text: str, name: str, replacement: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None)
            if end is None:
                raise RuntimeError(f"Python AST did not expose end_lineno for {name!r}")
            repl_lines = replacement.strip("\n").splitlines()
            return "\n".join(lines[:start] + repl_lines + lines[end:]) + "\n"
    raise RuntimeError(f"Could not find top-level function {name!r}")


REMOVE_BY_VALUES = r'''
def _remove_by_values(rows: list[dict[str, Any]], column: str, values: set[str]) -> list[dict[str, Any]]:
    """Return rows excluding entries whose column value is in values.

    This helper is deliberately row-oriented so the SIDRA denominator anchor does
    not bypass the production storage boundary with direct Polars parquet I/O.
    """
    if not values:
        return list(rows)
    return [dict(row) for row in rows if str(row.get(column)) not in values]
'''

APPEND_ROWS = r'''
def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str | None = None, remove_values: set[str] | None = None) -> None:
    """Append rows to an existing bundle table through output.table_io.

    Existing rows can be removed by a single key before append. Schema
    preservation is delegated to write_rows_like(), which itself delegates to
    the canonical pegasus.storage boundary.
    """
    existing = read_rows(path)
    if remove_column and remove_values:
        existing = _remove_by_values(existing, remove_column, {str(value) for value in remove_values})
    write_rows_like(path, existing + [dict(row) for row in rows])
'''

GET_FIELD = r'''
def _get_field_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one V_fields row named {name!r}, found {len(matches)}")
    return matches[0]
'''

AUDIT_SOURCE = r'''
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "src" / "pegasus" / "output" / "sidra_denominator_anchor.py"
BANNED = (
    r"\bpl\.read_parquet\b",
    r"\.write_parquet\b",
    r"\bpyarrow\.parquet\b",
    r"\bpq\.read_table\b",
    r"\bpq\.write_table\b",
)


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    errors: list[str] = []
    for pattern in BANNED:
        if re.search(pattern, text):
            errors.append(f"storage bypass remains in {TARGET.relative_to(ROOT)}: {pattern}")
    if "from pegasus.output.table_io import" not in text:
        errors.append("sidra_denominator_anchor.py does not import output.table_io")
    payload = {
        "audit": "slice28za_sidra_anchor_storage_boundary",
        "target": str(TARGET.relative_to(ROOT)),
        "errors": errors,
        "status": "failed" if errors else "passed",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

TEST_UNIT_SOURCE = r'''
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
    assert "from pegasus.output.table_io import" in text
'''

TEST_INT_SOURCE = r'''
from __future__ import annotations

import json
import subprocess
import sys


def test_slice28za_sidra_anchor_storage_audit_passes():
    result = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28za_sidra_anchor_storage_boundary.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
'''


def _patch_sidra_anchor() -> None:
    if not SIDRA_ANCHOR.exists():
        raise FileNotFoundError(SIDRA_ANCHOR)
    _backup(SIDRA_ANCHOR, ".slice28za_sidra_storage_boundary.bak")
    text = SIDRA_ANCHOR.read_text(encoding="utf-8")

    # The anchor should not need direct Polars parquet I/O after Slice 28ZA.
    text = text.replace("\nimport polars as pl\n", "\n")
    import_line = "from pegasus.output.table_io import read_rows, table_schema, write_rows_like\n"
    if import_line not in text:
        anchor = "from pegasus.output.validate import validate_output_bundle\n"
        if anchor not in text:
            raise RuntimeError("Could not locate output.validate import anchor")
        text = text.replace(anchor, anchor + import_line)

    text = _replace_top_level_function(text, "_remove_by_values", REMOVE_BY_VALUES)
    text = _replace_top_level_function(text, "_append_rows", APPEND_ROWS)
    text = _replace_top_level_function(text, "_get_field_by_name", GET_FIELD)

    replacements = {
        "v = pl.read_parquet(v_path)": "v = read_rows(v_path)",
        dedent('''
            v_clean = v.filter(~pl.col("name").is_in(sorted(new_names)))
            v_clean.write_parquet(v_path)
            _append_rows(v_path, [population_row, rate_row], remove_column="field_id", remove_values=new_field_ids)
        ''').strip(): dedent('''
            v_rows = _remove_by_values(v, "name", new_names)
            write_rows_like(v_path, v_rows)
            _append_rows(v_path, [population_row, rate_row], remove_column="field_id", remove_values=new_field_ids)
        ''').strip(),
        dedent('''
            e = pl.read_parquet(e_path)
            e = e.filter(pl.col("child_field_id") != rate_row["field_id"])
            e.write_parquet(e_path)
            _append_rows(e_path, edge_rows)
        ''').strip(): dedent('''
            _append_rows(e_path, edge_rows, remove_column="child_field_id", remove_values={rate_row["field_id"]})
        ''').strip(),
        dedent('''
            vd = pl.read_parquet(vd_path)
            vd_columns = vd.columns
            vd = vd.filter(~pl.col("field_id").is_in(sorted(new_field_ids)))
            vd.write_parquet(vd_path)
            _append_rows(vd_path, _vd_rows(vd_columns=vd_columns, population_row=population_row, rate_row=rate_row))
        ''').strip(): dedent('''
            vd_columns = list(table_schema(vd_path).names)
            _append_rows(
                vd_path,
                _vd_rows(vd_columns=vd_columns, population_row=population_row, rate_row=rate_row),
                remove_column="field_id",
                remove_values=new_field_ids,
            )
        ''').strip(),
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"Could not locate expected sidra_denominator_anchor block:\n{old}")
        text = text.replace(old, new)

    banned = ["pl.read_parquet", ".write_parquet", "pyarrow.parquet", "pq.read_table", "pq.write_table"]
    remaining = [token for token in banned if token in text]
    if remaining:
        raise RuntimeError(f"Storage bypass tokens remain in sidra_denominator_anchor.py: {remaining}")

    SIDRA_ANCHOR.write_text(text, encoding="utf-8")


def _write_auxiliary_files() -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    TEST_UNIT.parent.mkdir(parents=True, exist_ok=True)
    TEST_INT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(AUDIT_SOURCE.lstrip(), encoding="utf-8")
    TEST_UNIT.write_text(TEST_UNIT_SOURCE.lstrip(), encoding="utf-8")
    TEST_INT.write_text(TEST_INT_SOURCE.lstrip(), encoding="utf-8")
    if DOC.exists():
        doc = DOC.read_text(encoding="utf-8")
        note = "\n\nSlice 28ZA extends the storage-boundary adoption to the SIDRA denominator anchor. The module may still use row-level Python transformations, but Parquet reads/writes are routed through output.table_io and pegasus.storage.\n"
        if "Slice 28ZA extends the storage-boundary adoption" not in doc:
            DOC.write_text(doc.rstrip() + note, encoding="utf-8")


def main() -> None:
    if not (ROOT / "src" / "pegasus").exists():
        raise RuntimeError("Run this updater from the PegaSUS repository root")
    _patch_sidra_anchor()
    _write_auxiliary_files()
    print("Slice 28ZA SIDRA denominator anchor storage-boundary adoption applied.")


if __name__ == "__main__":
    main()
