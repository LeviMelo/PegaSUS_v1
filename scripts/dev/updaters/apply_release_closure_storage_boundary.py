from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "src/pegasus/output/validate.py"


def main() -> int:
    source = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parquet_imports = [
        node for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "pyarrow.parquet" for alias in node.names)
    ]
    read_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_read"
    ]
    if len(parquet_imports) != 1 or len(read_functions) != 1:
        raise RuntimeError(
            f"unexpected validator shape: parquet_imports={len(parquet_imports)} read_functions={len(read_functions)}"
        )
    if "from pegasus.storage import read_table" in source:
        print("release closure validator migration already applied")
        return 0
    lines = source.splitlines(keepends=True)
    import_node = parquet_imports[0]
    function_node = read_functions[0]
    replacements = [
        (import_node.lineno - 1, import_node.end_lineno, "from pegasus.storage import read_table\n"),
        (
            function_node.lineno - 1,
            function_node.end_lineno,
            "def _read(path: Path):\n    return read_table(path)\n",
        ),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    updated = "".join(lines)
    ast.parse(updated)
    if "pyarrow.parquet" in updated or "pq.read_table(" in updated:
        raise RuntimeError("validator storage-boundary post-audit failed")
    TARGET.write_text(updated, encoding="utf-8")
    print("migrated output validator reads to pegasus.storage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
