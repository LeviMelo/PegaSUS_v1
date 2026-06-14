from __future__ import annotations

from pathlib import Path
import re

REPO = Path.cwd()

READ_ONLY = REPO / "src" / "pegasus" / "dashboard" / "read_only.py"
UPDATER = REPO / "scripts" / "dev" / "updaters" / "apply_slice28z_storage_boundary_adoption.py"

NEW_READ_TABLE_HEAD = '''def read_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
    assert_no_compute_trigger("table_head")
    if limit < 0 or limit > 500:
        raise ValueError("Dashboard table head limit must be between 0 and 500.")
    root = _run_dir(run_dir)
    path = _table_path(root, table_name)
    rows = read_rows(path)[:limit]
    schema_obj = table_schema(path)
    return {
        "table": table_name,
        "path": TABLE_FILES[table_name],
        "rows_returned": len(rows),
        "row_count": parquet_row_count(path),
        "columns": list(schema_obj.names),
        "rows": rows,
    }
'''


def _backup(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _replace_top_level_function(text: str, name: str, replacement: str) -> str:
    pattern = re.compile(rf"^def {re.escape(name)}\([^\n]*\):\n(?:    .*\n|\n)*?(?=^def |^class |\Z)", re.MULTILINE)
    updated, count = pattern.subn(replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace top-level function {name!r}")
    return updated


def _patch_read_only() -> None:
    if not READ_ONLY.exists():
        raise FileNotFoundError(READ_ONLY)
    _backup(READ_ONLY, ".slice28z_dashboard_readonly_contract.bak")
    text = READ_ONLY.read_text(encoding="utf-8")

    # Keep the dashboard contract check on the canonical allowlist token.  The
    # Slice 28Z storage rewrite accidentally called an undefined/non-imported
    # assertion with the wrong operation name.
    text = text.replace('assert_read_only_operation("read_table_head")', 'assert_no_compute_trigger("table_head")')
    text = text.replace("assert_read_only_operation('read_table_head')", 'assert_no_compute_trigger("table_head")')

    table_io_import = "from pegasus.output.table_io import read_rows, table_row_count, table_schema"
    if re.search(r"^from pegasus\.output\.table_io import .*$", text, flags=re.MULTILINE):
        text = re.sub(
            r"^from pegasus\.output\.table_io import .*$",
            table_io_import,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        marker = "from pegasus.dashboard.contracts import assert_no_compute_trigger\n"
        if marker in text:
            text = text.replace(marker, marker + table_io_import + "\n", 1)
        else:
            raise RuntimeError("Could not find dashboard contracts import anchor")

    text = _replace_top_level_function(text, "read_table_head", NEW_READ_TABLE_HEAD)
    READ_ONLY.write_text(text, encoding="utf-8")


def _patch_updater() -> None:
    if not UPDATER.exists():
        return
    _backup(UPDATER, ".slice28z_dashboard_readonly_contract.bak")
    text = UPDATER.read_text(encoding="utf-8")
    text = text.replace(
        "from pegasus.output.table_io import read_rows, table_row_count",
        "from pegasus.output.table_io import read_rows, table_row_count, table_schema",
    )
    text = text.replace(
        "from pegasus.output.table_io import read_rows, table_row_count, table_schema, table_schema",
        "from pegasus.output.table_io import read_rows, table_row_count, table_schema",
    )
    text = text.replace('assert_read_only_operation("read_table_head")', 'assert_no_compute_trigger("table_head")')
    text = text.replace("assert_read_only_operation('read_table_head')", 'assert_no_compute_trigger("table_head")')

    # If the updater contains the simplified Slice 28Z read_table_head replacement,
    # make future re-runs generate the compatibility-preserving implementation.
    if "def read_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:" in text:
        try:
            text = _replace_top_level_function(text, "read_table_head", NEW_READ_TABLE_HEAD)
        except RuntimeError:
            # Updater may store this inside a dedented string rather than as an
            # executable top-level function.  In that case the string replacements
            # above are still sufficient for the actual repo repair.
            pass

    # Patch the common dedent string body shape if present.
    old_fragment = '''def read_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
    assert_no_compute_trigger("table_head")
    root = _run_dir(run_dir)
    path = _table_path(root, table_name)
    rows = read_rows(path)[:limit]
    return {
        "table": table_name,
        "path": TABLE_FILES[table_name],
        "rows": rows,
    }
'''
    if old_fragment in text:
        text = text.replace(old_fragment, NEW_READ_TABLE_HEAD)

    UPDATER.write_text(text, encoding="utf-8")


def main() -> None:
    _patch_read_only()
    _patch_updater()
    print("Slice 28Z dashboard read-only contract repair applied.")
    print("Patched:")
    print(f"- {READ_ONLY}")
    if UPDATER.exists():
        print(f"- {UPDATER}")


if __name__ == "__main__":
    main()
