
from __future__ import annotations

import ast
from pathlib import Path


def _top_level_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_slice15d_cli_commands_are_declared_once() -> None:
    path = Path("src/pegasus/cli.py")
    text = path.read_text(encoding="utf-8")
    assert text.count('@efg_app.command("plan-promotion")') == 1
    assert text.count('@efg_app.command("attach-promotion-plan")') == 1
    assert text.count('@efg_app.command("apply-promotion-plan")') == 1
    assert text.count('@efg_app.command("inspect-promotion-plan")') == 1
    assert _top_level_defs(path, "efg_plan_promotion") == 1
    assert _top_level_defs(path, "efg_attach_promotion_plan") == 1
    assert _top_level_defs(path, "efg_apply_promotion_plan") == 1
    assert _top_level_defs(path, "efg_inspect_promotion_plan") == 1


def test_slice15d_compile_boundary_remains_consolidated() -> None:
    path = Path("src/pegasus/workflows/compile.py")
    assert _top_level_defs(path, "run_compile") == 1
    assert _top_level_defs(path, "_run_compile_impl") == 1


def test_slice15d_workflow_api_is_present() -> None:
    promotion = Path("src/pegasus/workflows/efg_promotion.py").read_text(encoding="utf-8")
    apply = Path("src/pegasus/workflows/efg_apply.py").read_text(encoding="utf-8")
    assert "run_plan_efg_promotion" in promotion
    assert "run_attach_efg_promotion_plan_to_run" in promotion
    assert "run_apply_efg_promotion_plan" in apply
