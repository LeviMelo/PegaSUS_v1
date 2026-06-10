import ast
from pathlib import Path


def test_cli_uses_workflow_wrappers_for_mutating_slice_commands():
    tree = ast.parse(Path("src/pegasus/cli.py").read_text(encoding="utf-8"))
    imported_from = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_from.append(node.module)

    forbidden = {
        "pegasus.sidra.extract",
        "pegasus.sidra.metadata",
        "pegasus.sidra.plan",
        "pegasus.sidra.registry",
        "pegasus.sidra.facts",
        "pegasus.datasus.normalize",
        "pegasus.datasus.manifests",
        "pegasus.datasus.subprocess",
        "pegasus.output.sidra_denominator_anchor",
    }
    assert not (set(imported_from) & forbidden)
    assert "pegasus.workflows.sidra" in imported_from
    assert "pegasus.workflows.datasus" in imported_from
    assert "pegasus.workflows.efg" in imported_from
