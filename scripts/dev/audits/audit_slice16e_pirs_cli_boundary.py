
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner


def _top_level_function_count(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    cli_path = Path("src/pegasus/cli.py")
    compile_path = Path("src/pegasus/workflows/compile.py")
    cli = cli_path.read_text(encoding="utf-8")
    compile_text = compile_path.read_text(encoding="utf-8")

    if cli.count("# ---- Slice 16E PIRS planning CLI boundary ----") != 1:
        errors.append("Slice 16E CLI marker must appear exactly once")
    for token in (
        '@pirs_app.command("candidates-from-run")',
        '@pirs_app.command("attach-candidate-gate")',
        '@pirs_app.command("plan-selection")',
        '@pirs_app.command("attach-selection-plan")',
        '@pirs_app.command("plan-design")',
        '@pirs_app.command("attach-design-plan")',
        '@pirs_app.command("check-design-readiness")',
        '@pirs_app.command("attach-design-readiness")',
        '@pirs_app.command("inspect-candidates")',
        '@pirs_app.command("inspect-selection-plan")',
        '@pirs_app.command("inspect-design-plan")',
        '@pirs_app.command("inspect-design-readiness")',
    ):
        if token not in cli:
            errors.append(f"missing CLI token: {token}")
    if _top_level_function_count(compile_text, "run_compile") != 1:
        errors.append("compile.py public run_compile count changed")
    if _top_level_function_count(compile_text, "_run_compile_impl") != 1:
        errors.append("compile.py private _run_compile_impl count changed")

    from pegasus.cli import app

    runner = CliRunner()
    help_result = runner.invoke(app, ["pirs", "--help"])
    if help_result.exit_code != 0:
        errors.append("pirs --help failed")
    else:
        for command in ("candidates-from-run", "plan-selection", "plan-design", "check-design-readiness"):
            if command not in help_result.output:
                errors.append(f"pirs --help missing command: {command}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidates.json"
        path.write_text(json.dumps({"gate": "pirs_candidate_gate", "candidate_count": 1, "rejected_count": 0, "candidates": [{}], "rejected": []}), encoding="utf-8")
        result = runner.invoke(app, ["pirs", "inspect-candidates", "--manifest", str(path)])
        if result.exit_code != 0 or '"candidate_count": 1' not in result.output:
            errors.append("inspect-candidates command failed on minimal manifest")

    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16E PIRS planning CLI boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
