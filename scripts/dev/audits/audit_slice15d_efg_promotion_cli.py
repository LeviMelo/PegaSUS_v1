
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    cli_path = Path("src/pegasus/cli.py")
    cli_text = cli_path.read_text(encoding="utf-8")
    for command in (
        '@efg_app.command("plan-promotion")',
        '@efg_app.command("attach-promotion-plan")',
        '@efg_app.command("apply-promotion-plan")',
        '@efg_app.command("inspect-promotion-plan")',
    ):
        if cli_text.count(command) != 1:
            errors.append(f"CLI command declaration count is not exactly one: {command}")
    for name in (
        "efg_plan_promotion",
        "efg_attach_promotion_plan",
        "efg_apply_promotion_plan",
        "efg_inspect_promotion_plan",
    ):
        if _count_defs(cli_path, name) != 1:
            errors.append(f"CLI function definition count is not exactly one: {name}")
    compile_path = Path("src/pegasus/workflows/compile.py")
    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must contain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must contain exactly one private _run_compile_impl")

    from pegasus.cli import app  # noqa: F401
    from pegasus.workflows.construct.efg_promotion import run_attach_efg_promotion_plan_to_run, run_plan_efg_promotion
    from pegasus.workflows.construct.efg_apply import run_apply_efg_promotion_plan

    if "run_dir" not in inspect.signature(run_plan_efg_promotion).parameters:
        errors.append("run_plan_efg_promotion missing run_dir parameter")
    if "materialization_manifest" not in inspect.signature(run_attach_efg_promotion_plan_to_run).parameters:
        errors.append("run_attach_efg_promotion_plan_to_run missing materialization_manifest parameter")
    if "promotion_plan" not in inspect.signature(run_apply_efg_promotion_plan).parameters:
        errors.append("run_apply_efg_promotion_plan missing promotion_plan parameter")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 15D EFG promotion CLI boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
