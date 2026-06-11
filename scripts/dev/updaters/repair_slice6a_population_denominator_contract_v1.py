from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "src" / "pegasus" / "she" / "population" / "schema.py"


OFFICIAL_FN = """def official_sidra_anchor_contract(
    *,
    source: str = "SIDRA",
    warnings: list[str] | None = None,
) -> DenominatorContract:
    return DenominatorContract(
        mode="official_sidra_anchor",
        source=source,
        provenance=["official"],
        state="fragile",
        dashboard_safe="warning",
        allowed_for_rates=True,
        warnings=warnings or ["fixture_or_unvalidated_sidra_anchor"],
    )
"""


BLOCKED_FN = """def blocked_missing_population_contract(
    *,
    reason: str,
) -> DenominatorContract:
    return DenominatorContract(
        mode="blocked_missing",
        source="none",
        provenance=[],
        state="illegal_excluded",
        dashboard_safe=False,
        allowed_for_rates=False,
        warnings=[reason],
    )
"""


def _replace_function(source: str, function_name: str, replacement: str) -> str:
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]
    if len(nodes) != 1:
        raise RuntimeError(f"Expected exactly one top-level {function_name}() in {SCHEMA}; found {len(nodes)}.")

    node = nodes[0]
    if not hasattr(node, "end_lineno") or node.end_lineno is None:
        raise RuntimeError("Python AST did not expose end_lineno; cannot patch safely.")

    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno

    replacement_text = replacement.rstrip() + "\n"
    return "".join(lines[:start]) + replacement_text + "".join(lines[end:])


def patch_schema() -> None:
    if not SCHEMA.exists():
        raise RuntimeError(f"Missing expected file: {SCHEMA}")

    source = SCHEMA.read_text(encoding="utf-8")
    if "class PopulationTensorRequest" not in source:
        raise RuntimeError("Slice 6A schema.py shape not detected: PopulationTensorRequest missing.")

    patched = _replace_function(source, "official_sidra_anchor_contract", OFFICIAL_FN)
    patched = _replace_function(patched, "blocked_missing_population_contract", BLOCKED_FN)

    while "\n\n\n\n" in patched:
        patched = patched.replace("\n\n\n\n", "\n\n\n")

    SCHEMA.write_text(patched, encoding="utf-8")


def main() -> None:
    print("Slice 6A denominator contract repair preflight passed.")
    print("PATCH:")
    print("  src/pegasus/she/population/schema.py")
    patch_schema()
    print("Applied Slice 6A denominator contract repair.")


if __name__ == "__main__":
    main()
