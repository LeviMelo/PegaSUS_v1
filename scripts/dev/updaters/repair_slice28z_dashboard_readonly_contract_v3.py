from __future__ import annotations

import ast
import shutil
from pathlib import Path

ROOT = Path.cwd()
READ_ONLY = ROOT / "src" / "pegasus" / "dashboard" / "read_only.py"
APPLY_UPDATER = ROOT / "scripts" / "dev" / "updaters" / "apply_slice28z_storage_boundary_adoption.py"

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
        "row_count": table_row_count(path),
        "columns": list(schema_obj.names),
        "rows": rows,
    }
'''


def _replace_function_by_ast(text: str, function_name: str, replacement: str) -> str:
    """Replace a top-level function using AST line numbers, independent of formatting."""
    tree = ast.parse(text)
    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            target = node
            break
    if target is None or target.end_lineno is None:
        raise RuntimeError(f"Could not locate top-level function {function_name!r} by AST")

    lines = text.splitlines(keepends=True)
    start = target.lineno - 1
    end = target.end_lineno
    newline = "\r\n" if "\r\n" in text else "\n"
    replacement_text = replacement.rstrip("\n").replace("\n", newline) + newline
    lines[start:end] = [replacement_text]
    return "".join(lines)


def _ensure_single_line_import(text: str, module: str, names: list[str]) -> str:
    """Ensure a simple import line exists. If a single-line import exists, normalize it."""
    wanted = f"from {module} import {', '.join(names)}"
    lines = text.splitlines()
    prefix = f"from {module} import "
    changed = False
    for i, line in enumerate(lines):
        if line.startswith(prefix) and "(" not in line:
            existing = [item.strip() for item in line[len(prefix):].split(",") if item.strip()]
            merged = []
            for item in existing + names:
                if item not in merged:
                    merged.append(item)
            lines[i] = f"from {module} import {', '.join(merged)}"
            changed = True
            break
    if changed:
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    if wanted in text:
        return text

    # Insert after future imports and other imports near the top.
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("from __future__ import "):
            insert_at = i + 1
            continue
        if i >= insert_at and (line.startswith("import ") or line.startswith("from ")):
            insert_at = i + 1
            continue
        if i > insert_at and line.strip() == "":
            continue
        if i > insert_at:
            break
    lines.insert(insert_at, wanted)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _ensure_dashboard_contract_import(text: str) -> str:
    if "assert_no_compute_trigger" in text:
        return text
    return _ensure_single_line_import(
        text,
        "pegasus.dashboard.contracts",
        ["assert_no_compute_trigger"],
    )


def _ensure_table_io_import(text: str) -> str:
    return _ensure_single_line_import(
        text,
        "pegasus.output.table_io",
        ["read_rows", "table_row_count", "table_schema"],
    )


def _patch_read_only() -> None:
    if not READ_ONLY.exists():
        raise FileNotFoundError(READ_ONLY)
    text = READ_ONLY.read_text(encoding="utf-8")
    backup = READ_ONLY.with_suffix(READ_ONLY.suffix + ".slice28z_dashboard_readonly_contract_v3.bak")
    if not backup.exists():
        shutil.copy2(READ_ONLY, backup)

    text = _ensure_dashboard_contract_import(text)
    text = _ensure_table_io_import(text)
    text = _replace_function_by_ast(text, "read_table_head", NEW_READ_TABLE_HEAD)
    READ_ONLY.write_text(text, encoding="utf-8")


def _patch_apply_updater_best_effort() -> None:
    """Keep the original updater from reintroducing the narrowed dashboard contract.

    This is best-effort; runtime correctness is in src/pegasus/dashboard/read_only.py.
    """
    if not APPLY_UPDATER.exists():
        return
    text = APPLY_UPDATER.read_text(encoding="utf-8")
    backup = APPLY_UPDATER.with_suffix(APPLY_UPDATER.suffix + ".slice28z_dashboard_readonly_contract_v3.bak")
    if not backup.exists():
        shutil.copy2(APPLY_UPDATER, backup)

    # Minimal textual normalization; do not fail the repair if the updater body differs.
    text = text.replace(
        "from pegasus.output.table_io import read_rows, table_row_count",
        "from pegasus.output.table_io import read_rows, table_row_count, table_schema",
    )
    text = text.replace(
        "assert_read_only_operation(\"read_table_head\")",
        "assert_no_compute_trigger(\"table_head\")",
    )
    text = text.replace(
        "assert_read_only_operation(\"table_head\")",
        "assert_no_compute_trigger(\"table_head\")",
    )
    if "rows_returned" not in text and "def read_table_head" in text:
        # Do not attempt a brittle rewrite of nested updater template blocks.
        # The applied source file is authoritative; this marker helps reviewers.
        text += "\n# slice28z repair v3: dashboard/read_only.py restored table_head contract via AST patcher.\n"
    APPLY_UPDATER.write_text(text, encoding="utf-8")


def main() -> None:
    _patch_read_only()
    _patch_apply_updater_best_effort()
    print("Slice 28Z dashboard read-only contract repair v3 applied.")
    print("Patched:", READ_ONLY)
    if APPLY_UPDATER.exists():
        print("Best-effort patched updater:", APPLY_UPDATER)


if __name__ == "__main__":
    main()
