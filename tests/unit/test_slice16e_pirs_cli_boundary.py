
from __future__ import annotations

from pathlib import Path


COMMAND_TOKENS = (
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
)


def test_slice16e_cli_exposes_pirs_planning_commands_without_compile_mutation() -> None:
    cli = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
    compile_text = Path("src/pegasus/workflows/compile.py").read_text(encoding="utf-8")

    assert cli.count("# ---- Slice 16E PIRS planning CLI boundary ----") == 1
    assert "def _pirs_cli_json" in cli
    assert "def _pirs_cli_print" in cli
    for token in COMMAND_TOKENS:
        assert token in cli

    assert compile_text.count("def run_compile(") == 1
    assert "def _run_compile_impl(" in compile_text
