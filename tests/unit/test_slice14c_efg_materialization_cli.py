
from __future__ import annotations

import ast
from pathlib import Path


def _top_level_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_slice14c_cli_commands_are_declared_once() -> None:
    path = Path("src/pegasus/cli.py")
    text = path.read_text(encoding="utf-8")
    assert text.count('@efg_app.command("materialize-substrate-manifest")') == 1
    assert text.count('@efg_app.command("attach-materialization")') == 1
    assert text.count('@efg_app.command("inspect-materialization")') == 1
    assert _top_level_defs(path, "efg_materialize_substrate_manifest") == 1
    assert _top_level_defs(path, "efg_attach_materialization_manifest") == 1
    assert _top_level_defs(path, "efg_inspect_materialization_manifest") == 1


def test_slice14c_compile_boundary_remains_consolidated() -> None:
    path = Path("src/pegasus/workflows/compile.py")
    assert _top_level_defs(path, "run_compile") == 1
    assert _top_level_defs(path, "_run_compile_impl") == 1


def test_slice14c_workflow_api_is_present() -> None:
    from pegasus.workflows.construct.efg_materialize import (
        run_attach_efg_materialization_to_run,
        run_materialize_substrate_manifest,
    )

    assert callable(run_materialize_substrate_manifest)
    assert callable(run_attach_efg_materialization_to_run)
