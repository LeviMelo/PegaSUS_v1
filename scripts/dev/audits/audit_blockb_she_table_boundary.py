from __future__ import annotations

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
