from __future__ import annotations

import ast
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TARGET_RELATIVE_PATHS = [
    "src/pegasus/she/maternal_child_linkage.py",
    "src/pegasus/she/sih_costs.py",
    "src/pegasus/she/population/sidra_anchor.py",
    "src/pegasus/she/stdfm/artifacts.py",
]

AUDIT_RELATIVE_PATH = "scripts/dev/audits/audit_blockb_she_table_boundary.py"
UNIT_TEST_RELATIVE_PATH = "tests/unit/test_blockb_she_table_boundary_source.py"
INTEGRATION_TEST_RELATIVE_PATH = "tests/integration/test_blockb_she_table_boundary_audit.py"
UPDATER_RELATIVE_PATH = "scripts/dev/updaters/apply_blockb_she_table_boundary.py"

MAX_ERROR_EXAMPLES = 20
BACKUP_SUFFIX = ".blockb_she_table_boundary.bak"


@dataclass(frozen=True)
class ForbiddenCall:
    kind: Literal["read_parquet", "write_parquet", "pyarrow_parquet"]
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    text: str


class PatchError(RuntimeError):
    """Raised when the Block B updater cannot safely patch the source tree."""


def find_repo_root() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path.cwd())
    except OSError:
        pass
    candidates.extend(Path(__file__).resolve().parents)

    for candidate in candidates:
        if (
            (candidate / "src" / "pegasus").exists()
            and (candidate / "tests").exists()
            and (candidate / "pyproject.toml").exists()
        ):
            return candidate.resolve()

    raise PatchError(
        "Could not locate repo root. Run this updater from the PegaSUS repo root "
        "or place it under scripts/dev/updaters."
    )


ROOT = find_repo_root()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _absolute_span(source: str, node: ast.AST) -> tuple[int, int]:
    if (
        getattr(node, "lineno", None) is None
        or getattr(node, "col_offset", None) is None
        or getattr(node, "end_lineno", None) is None
        or getattr(node, "end_col_offset", None) is None
    ):
        raise PatchError("AST node lacks source coordinates; cannot patch safely.")
    offsets = _line_offsets(source)
    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset
    return start, end


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment.strip()
    start, end = _absolute_span(source, node)
    return source[start:end].strip()


class ForbiddenCallFinder(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: list[tuple[ForbiddenCall, ast.Call]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            func = node.func

            if (
                isinstance(func.value, ast.Name)
                and func.value.id == "pl"
                and func.attr == "read_parquet"
            ):
                self.calls.append((self._call("read_parquet", node), node))

            elif func.attr == "write_parquet":
                self.calls.append((self._call("write_parquet", node), node))

            elif (
                isinstance(func.value, ast.Name)
                and func.value.id == "pq"
                and func.attr in {"read_table", "write_table"}
            ):
                self.calls.append((self._call("pyarrow_parquet", node), node))

        self.generic_visit(node)

    def _call(
        self,
        kind: Literal["read_parquet", "write_parquet", "pyarrow_parquet"],
        node: ast.Call,
    ) -> ForbiddenCall:
        return ForbiddenCall(
            kind=kind,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=node.end_lineno,
            end_col_offset=node.end_col_offset,
            text=_source_segment(self.source, node),
        )


def find_forbidden_calls(source: str) -> list[tuple[ForbiddenCall, ast.Call]]:
    tree = ast.parse(source)
    finder = ForbiddenCallFinder(source)
    finder.visit(tree)
    return finder.calls


def has_storage_import(source: str, imported_name: str) -> bool:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "pegasus.storage":
            if any(alias.name == imported_name for alias in node.names):
                return True
    return False


def import_insert_line(source: str) -> int:
    tree = ast.parse(source)
    insert_after = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            insert_after = max(insert_after, node.end_lineno or 0)
            continue

        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_after = max(insert_after, node.end_lineno or 0)
            continue

        break
    return insert_after


def insert_storage_import(source: str, imported_name: str) -> str:
    if has_storage_import(source, imported_name):
        return source

    lines = source.splitlines()
    idx = import_insert_line(source)
    import_line = f"from pegasus.storage import {imported_name}"

    if import_line in lines:
        return source

    lines = lines[:idx] + [import_line] + lines[idx:]
    return "\n".join(lines) + "\n"


def build_replacement(call: ForbiddenCall, node: ast.Call) -> tuple[str, set[str]]:
    if call.kind == "pyarrow_parquet":
        raise PatchError(
            f"Unsupported PyArrow parquet call at line {call.lineno}: {call.text}. "
            "Migrate this call manually or add an explicit safe rule."
        )

    if call.kind == "read_parquet":
        if len(node.args) != 1 or node.keywords:
            raise PatchError(
                f"Unsupported pl.read_parquet shape at line {call.lineno}: {call.text}. "
                "Only pl.read_parquet(path) is automatically migrated."
            )
        path_expr = ast.unparse(node.args[0])
        return f"pl.from_arrow(read_table({path_expr}))", {"read_table"}

    if call.kind == "write_parquet":
        if len(node.args) != 1 or node.keywords:
            raise PatchError(
                f"Unsupported write_parquet shape at line {call.lineno}: {call.text}. "
                "Only df.write_parquet(path) is automatically migrated."
            )

        if not isinstance(node.func, ast.Attribute):
            raise PatchError(f"Unexpected write_parquet call shape at line {call.lineno}: {call.text}")

        path_expr = ast.unparse(node.args[0])
        receiver_expr = ast.unparse(node.func.value)
        return f"write_table({path_expr}, {receiver_expr}.to_arrow())", {"write_table"}

    raise PatchError(f"Unhandled call kind: {call.kind}")


def patch_source(source: str, path: Path) -> tuple[str, dict[str, Any]]:
    calls = find_forbidden_calls(source)

    if not calls:
        return source, {
            "path": _rel(path),
            "pre_patch_forbidden_calls": 0,
            "changed": False,
        }

    replacements: list[tuple[int, int, str]] = []
    imports_needed: set[str] = set()
    inventory: list[dict[str, Any]] = []

    for call, node in calls:
        replacement, imports = build_replacement(call, node)
        start, end = _absolute_span(source, node)
        replacements.append((start, end, replacement))
        imports_needed.update(imports)
        inventory.append(
            {
                "kind": call.kind,
                "line": call.lineno,
                "text": call.text,
            }
        )

    replacements.sort(key=lambda item: item[0], reverse=True)

    patched = source
    for start, end, replacement in replacements:
        patched = patched[:start] + replacement + patched[end:]

    for imported_name in sorted(imports_needed):
        patched = insert_storage_import(patched, imported_name)

    try:
        ast.parse(patched)
    except SyntaxError as exc:
        raise PatchError(f"Generated invalid Python for {_rel(path)}: {exc}") from exc

    remaining = find_forbidden_calls(patched)
    if remaining:
        examples = [
            {
                "kind": call.kind,
                "line": call.lineno,
                "text": call.text,
            }
            for call, _node in remaining[:MAX_ERROR_EXAMPLES]
        ]
        raise PatchError(
            f"Self-audit failed for {_rel(path)}; forbidden calls remain: "
            + json.dumps(examples, ensure_ascii=False)
        )

    return patched, {
        "path": _rel(path),
        "pre_patch_forbidden_calls": len(calls),
        "changed": patched != source,
        "inventory": inventory[:MAX_ERROR_EXAMPLES],
    }


AUDIT_CODE = r'''from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

TARGET_RELATIVE_PATHS = [
    "src/pegasus/she/maternal_child_linkage.py",
    "src/pegasus/she/sih_costs.py",
    "src/pegasus/she/population/sidra_anchor.py",
    "src/pegasus/she/stdfm/artifacts.py",
]

MAX_EXAMPLES = 20


def _segment(source: str, node: ast.AST) -> str:
    return (ast.get_source_segment(source, node) or "").strip()


class ForbiddenCallFinder(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.violations: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            func = node.func

            if (
                isinstance(func.value, ast.Name)
                and func.value.id == "pl"
                and func.attr == "read_parquet"
            ):
                self.violations.append(
                    {
                        "kind": "read_parquet",
                        "line": node.lineno,
                        "text": _segment(self.source, node),
                    }
                )

            elif func.attr == "write_parquet":
                self.violations.append(
                    {
                        "kind": "write_parquet",
                        "line": node.lineno,
                        "text": _segment(self.source, node),
                    }
                )

            elif (
                isinstance(func.value, ast.Name)
                and func.value.id == "pq"
                and func.attr in {"read_table", "write_table"}
            ):
                self.violations.append(
                    {
                        "kind": "pyarrow_parquet",
                        "line": node.lineno,
                        "text": _segment(self.source, node),
                    }
                )

        self.generic_visit(node)


def scan_file(path: Path) -> list[dict[str, Any]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    finder = ForbiddenCallFinder(source)
    finder.visit(tree)
    return finder.violations


def run_audit() -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    checked_files: list[str] = []

    for rel in TARGET_RELATIVE_PATHS:
        path = ROOT / rel
        if not path.exists():
            continue

        checked_files.append(rel)
        try:
            for violation in scan_file(path):
                errors.append({"path": rel, **violation})
        except Exception as exc:
            errors.append(
                {
                    "path": rel,
                    "kind": "audit_error",
                    "line": None,
                    "text": str(exc),
                }
            )

    return {
        "audit": "blockb_she_table_boundary",
        "status": "passed" if not errors else "failed",
        "checked_files": checked_files,
        "error_count": len(errors),
        "errors": errors[:MAX_EXAMPLES],
        "truncated": len(errors) > MAX_EXAMPLES,
    }


def main() -> None:
    result = run_audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
'''


UNIT_TEST_CODE = r'''from __future__ import annotations

from scripts.dev.audits.audit_blockb_she_table_boundary import run_audit


def test_blockb_she_table_boundary_source_is_clean() -> None:
    result = run_audit()
    assert result["status"] == "passed", result
    assert result["error_count"] == 0, result
'''


INTEGRATION_TEST_CODE = r'''from __future__ import annotations

from scripts.dev.audits.audit_blockb_she_table_boundary import run_audit


def test_blockb_she_table_boundary_audit_public_api_passes() -> None:
    result = run_audit()
    assert result["audit"] == "blockb_she_table_boundary"
    assert result["status"] == "passed", result
    assert isinstance(result["checked_files"], list)
'''


def write_text_file(path: Path, text: str, backup_suffix: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == text:
        return False

    if path.exists():
        backup = path.with_suffix(path.suffix + backup_suffix)
        if not backup.exists():
            backup.write_text(old or "", encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def install_self() -> bool:
    destination = ROOT / UPDATER_RELATIVE_PATH
    current = Path(__file__).resolve()
    if current == destination.resolve():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    new_text = current.read_text(encoding="utf-8")
    old_text = destination.read_text(encoding="utf-8") if destination.exists() else None
    if old_text == new_text:
        return False

    if destination.exists():
        backup = destination.with_suffix(destination.suffix + BACKUP_SUFFIX)
        if not backup.exists():
            backup.write_text(old_text or "", encoding="utf-8")

    destination.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    report: dict[str, Any] = {
        "block": "Block B",
        "status": "failed",
        "repo_root": str(ROOT),
        "source_files": [],
        "generated_files": [],
    }

    try:
        patched_sources: dict[Path, str] = {}
        source_reports: list[dict[str, Any]] = []

        for rel in TARGET_RELATIVE_PATHS:
            path = ROOT / rel
            if not path.exists():
                source_reports.append(
                    {
                        "path": rel,
                        "missing": True,
                        "changed": False,
                        "pre_patch_forbidden_calls": 0,
                    }
                )
                continue

            source = path.read_text(encoding="utf-8")
            patched, item_report = patch_source(source, path)
            patched_sources[path] = patched
            source_reports.append(item_report)

        # Cross-file validation before any production-source write.
        for path, patched in patched_sources.items():
            remaining = find_forbidden_calls(patched)
            if remaining:
                raise PatchError(f"Forbidden calls remain in {_rel(path)} after in-memory patch.")

        modified_sources: list[str] = []
        for path, patched in patched_sources.items():
            current = path.read_text(encoding="utf-8")
            if current != patched:
                backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
                if not backup.exists():
                    backup.write_text(current, encoding="utf-8")
                path.write_text(patched, encoding="utf-8")
                modified_sources.append(_rel(path))

        generated: list[str] = []

        audit_path = ROOT / AUDIT_RELATIVE_PATH
        if write_text_file(audit_path, AUDIT_CODE, BACKUP_SUFFIX):
            generated.append(AUDIT_RELATIVE_PATH)

        unit_path = ROOT / UNIT_TEST_RELATIVE_PATH
        if write_text_file(unit_path, UNIT_TEST_CODE, BACKUP_SUFFIX):
            generated.append(UNIT_TEST_RELATIVE_PATH)

        integration_path = ROOT / INTEGRATION_TEST_RELATIVE_PATH
        if write_text_file(integration_path, INTEGRATION_TEST_CODE, BACKUP_SUFFIX):
            generated.append(INTEGRATION_TEST_RELATIVE_PATH)

        if install_self():
            generated.append(UPDATER_RELATIVE_PATH)

        # Final parse validation for generated files and changed source files.
        for rel in (AUDIT_RELATIVE_PATH, UNIT_TEST_RELATIVE_PATH, INTEGRATION_TEST_RELATIVE_PATH):
            ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for path in patched_sources:
            ast.parse(path.read_text(encoding="utf-8"))

        # Final scoped audit after writes.
        namespace: dict[str, Any] = {}
        exec((ROOT / AUDIT_RELATIVE_PATH).read_text(encoding="utf-8"), namespace)
        audit_result = namespace["run_audit"]()
        if audit_result.get("status") != "passed":
            raise PatchError("Final Block B audit failed: " + json.dumps(audit_result, ensure_ascii=False))

        report.update(
            {
                "status": "success",
                "source_files": source_reports,
                "modified_source_files": modified_sources,
                "generated_files": generated,
                "audit": audit_result,
            }
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0)

    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
