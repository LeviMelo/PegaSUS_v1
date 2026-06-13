from __future__ import annotations

import ast
import py_compile
import sys
import zipfile
from pathlib import Path
from textwrap import dedent

SLICE = "16E"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/cli.py",
    "tests/unit/test_slice16e_pirs_cli_boundary.py",
    "tests/integration/test_slice16e_cli_pirs_planning.py",
    "scripts/dev/audits/audit_slice16e_pirs_cli_boundary.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/",
    "src/pegasus/output/",
    "src/pegasus/efg/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)

MARKER_START = "# ---- Slice 16E PIRS planning CLI boundary ----"
MARKER_END = "# ---- End Slice 16E PIRS planning CLI boundary ----"


def fail(message: str) -> None:
    raise SystemExit(f"[slice16e] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if path.startswith(FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_function_count(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/cli.py",
        "src/pegasus/workflows/pirs_candidates.py",
        "src/pegasus/workflows/pirs_selection.py",
        "src/pegasus/workflows/pirs_design.py",
        "src/pegasus/workflows/pirs_readiness.py",
        "src/pegasus/pirs/run_candidates.py",
        "src/pegasus/pirs/selection_plan.py",
        "src/pegasus/pirs/design_plan.py",
        "src/pegasus/pirs/design_readiness.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    cli = read("src/pegasus/cli.py")
    if "pirs_app = typer.Typer" not in cli or 'app.add_typer(pirs_app, name="pirs")' not in cli:
        fail("cli.py does not expose pirs_app Typer boundary")
    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_function_count(compile_text, "run_compile") != 1 or top_level_function_count(compile_text, "_run_compile_impl") != 1:
        fail("compile.py public/private run_compile boundary is not in the Slice 13C shape")
    checks = {
        "src/pegasus/workflows/pirs_candidates.py": ("run_build_pirs_candidates_from_run", "run_attach_pirs_candidate_gate"),
        "src/pegasus/workflows/pirs_selection.py": ("run_write_pirs_selection_plan", "run_attach_pirs_selection_plan"),
        "src/pegasus/workflows/pirs_design.py": ("run_plan_pirs_design", "run_attach_pirs_design_plan_to_run"),
        "src/pegasus/workflows/pirs_readiness.py": ("run_write_pirs_design_readiness_manifest", "run_attach_pirs_design_readiness_to_run"),
    }
    for path, tokens in checks.items():
        text = read(path)
        for token in tokens:
            if token not in text:
                fail(f"{path} missing workflow API: {token}")


def strip_existing_block(text: str) -> str:
    start = text.find(MARKER_START)
    if start == -1:
        return text
    end = text.find(MARKER_END, start)
    if end == -1:
        return text[:start].rstrip() + "\n"
    end_line = text.find("\n", end)
    if end_line == -1:
        return text[:start].rstrip() + "\n"
    return (text[:start] + text[end_line + 1:]).rstrip() + "\n"


def cli_block() -> str:
    return dedent(r'''

    # ---- Slice 16E PIRS planning CLI boundary ----
    def _pirs_cli_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise typer.BadParameter(f"missing JSON artifact: {path}") from exc
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"invalid JSON artifact: {path}: {exc}") from exc


    def _pirs_cli_print(payload: object) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


    @pirs_app.command("candidates-from-run")
    def pirs_candidates_from_run(
        run_dir: Path = typer.Option(..., "--run-dir"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        """Build a non-mutating PIRS field-candidate manifest from a completed run."""
        from pegasus.workflows.pirs_candidates import run_build_pirs_candidates_from_run

        _pirs_cli_print(run_build_pirs_candidates_from_run(run_dir=run_dir, output=output))


    @pirs_app.command("attach-candidate-gate")
    def pirs_attach_candidate_gate(
        run_dir: Path = typer.Option(..., "--run-dir"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        """Write and attach the PIRS candidate-gate summary to an existing run."""
        from pegasus.workflows.pirs_candidates import run_attach_pirs_candidate_gate

        _pirs_cli_print(run_attach_pirs_candidate_gate(run_dir=run_dir, output=output))


    @pirs_app.command("plan-selection")
    def pirs_plan_selection(
        run_dir: Path = typer.Option(..., "--run-dir"),
        candidate_manifest: Path | None = typer.Option(None, "--candidate-manifest"),
        output: Path | None = typer.Option(None, "--output"),
        budget: str = typer.Option("fast", "--budget"),
    ) -> None:
        """Build a non-mutating PIRS selection plan from candidate manifest data."""
        from pegasus.workflows.pirs_selection import run_write_pirs_selection_plan

        _pirs_cli_print(
            run_write_pirs_selection_plan(
                run_dir=run_dir,
                candidate_manifest=candidate_manifest,
                output=output,
                budget=budget,
            )
        )


    @pirs_app.command("attach-selection-plan")
    def pirs_attach_selection_plan(
        run_dir: Path = typer.Option(..., "--run-dir"),
        candidate_manifest: Path | None = typer.Option(None, "--candidate-manifest"),
        output: Path | None = typer.Option(None, "--output"),
        budget: str = typer.Option("fast", "--budget"),
    ) -> None:
        """Write and attach a PIRS selection-plan summary to an existing run."""
        from pegasus.workflows.pirs_selection import run_attach_pirs_selection_plan

        _pirs_cli_print(
            run_attach_pirs_selection_plan(
                run_dir=run_dir,
                candidate_manifest=candidate_manifest,
                output=output,
                budget=budget,
            )
        )


    @pirs_app.command("plan-design")
    def pirs_plan_design(
        run_dir: Path = typer.Option(..., "--run-dir"),
        selection_plan: Path | None = typer.Option(None, "--selection-plan"),
        output: Path | None = typer.Option(None, "--output"),
        budget: str | None = typer.Option(None, "--budget"),
    ) -> None:
        """Build a planned-only PIRS model-design contract from a selection plan."""
        from pegasus.workflows.pirs_design import run_plan_pirs_design

        _pirs_cli_print(
            run_plan_pirs_design(
                run_dir=run_dir,
                selection_plan=selection_plan,
                output=output,
                budget=budget,
            )
        )


    @pirs_app.command("attach-design-plan")
    def pirs_attach_design_plan(
        run_dir: Path = typer.Option(..., "--run-dir"),
        selection_plan: Path | None = typer.Option(None, "--selection-plan"),
        output: Path | None = typer.Option(None, "--output"),
        budget: str | None = typer.Option(None, "--budget"),
    ) -> None:
        """Write and attach a PIRS design-plan summary to an existing run."""
        from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run

        _pirs_cli_print(
            run_attach_pirs_design_plan_to_run(
                run_dir=run_dir,
                selection_plan=selection_plan,
                output=output,
                budget=budget,
            )
        )


    @pirs_app.command("check-design-readiness")
    def pirs_check_design_readiness(
        run_dir: Path = typer.Option(..., "--run-dir"),
        design_plan: Path | None = typer.Option(None, "--design-plan"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        """Write a design-readiness manifest without attaching it to run metadata."""
        from pegasus.workflows.pirs_readiness import run_write_pirs_design_readiness_manifest

        _pirs_cli_print(
            run_write_pirs_design_readiness_manifest(
                run_dir=run_dir,
                design_plan=design_plan,
                output=output,
            )
        )


    @pirs_app.command("attach-design-readiness")
    def pirs_attach_design_readiness(
        run_dir: Path = typer.Option(..., "--run-dir"),
        design_plan: Path | None = typer.Option(None, "--design-plan"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        """Write and attach the PIRS design-readiness gate to an existing run."""
        from pegasus.workflows.pirs_readiness import run_attach_pirs_design_readiness_to_run

        _pirs_cli_print(
            run_attach_pirs_design_readiness_to_run(
                run_dir=run_dir,
                design_plan=design_plan,
                output=output,
            )
        )


    @pirs_app.command("inspect-candidates")
    def pirs_inspect_candidates(
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        """Read an existing PIRS candidate manifest without mutating a run."""
        payload = _pirs_cli_json(manifest)
        _pirs_cli_print(
            {
                "artifact": payload.get("gate", "pirs_candidate_gate"),
                "candidate_count": payload.get("candidate_count", len(payload.get("candidates", []))),
                "rejected_count": payload.get("rejected_count", len(payload.get("rejected", []))),
                "manifest": str(manifest),
                "read_only": True,
            }
        )


    @pirs_app.command("inspect-selection-plan")
    def pirs_inspect_selection_plan(
        plan: Path = typer.Option(..., "--plan"),
    ) -> None:
        """Read an existing PIRS selection plan without mutating a run."""
        payload = _pirs_cli_json(plan)
        _pirs_cli_print(
            {
                "artifact": payload.get("artifact", "pirs_selection_plan"),
                "status": payload.get("status"),
                "budget": payload.get("budget"),
                "selected_outcome_field_id": payload.get("selected_outcome_field_id"),
                "selected_covariate_count": len(payload.get("selected_covariate_field_ids", [])),
                "selection_rejected_count": len(payload.get("selection_rejected", [])),
                "manifest": str(plan),
                "read_only": True,
            }
        )


    @pirs_app.command("inspect-design-plan")
    def pirs_inspect_design_plan(
        plan: Path = typer.Option(..., "--plan"),
    ) -> None:
        """Read an existing PIRS design plan without mutating a run."""
        payload = _pirs_cli_json(plan)
        _pirs_cli_print(
            {
                "artifact": payload.get("artifact", "pirs_design_plan"),
                "status": payload.get("status"),
                "design_matrix_state": payload.get("design_matrix_state"),
                "model_fit_state": payload.get("model_fit_state"),
                "residual_state": payload.get("residual_state"),
                "term_count": len(payload.get("terms", [])),
                "manifest": str(plan),
                "read_only": True,
            }
        )


    @pirs_app.command("inspect-design-readiness")
    def pirs_inspect_design_readiness(
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        """Read an existing PIRS design-readiness manifest without mutating a run."""
        payload = _pirs_cli_json(manifest)
        _pirs_cli_print(
            {
                "artifact": payload.get("artifact", "pirs_design_readiness_gate"),
                "status": payload.get("status"),
                "ready": payload.get("ready"),
                "accepted_field_count": len(payload.get("accepted_fields", [])),
                "rejected_field_count": len(payload.get("rejected_fields", [])),
                "manifest": str(manifest),
                "read_only": True,
            }
        )
    # ---- End Slice 16E PIRS planning CLI boundary ----
    ''')


def patch_cli() -> None:
    path = "src/pegasus/cli.py"
    text = strip_existing_block(read(path))
    if "import json" not in text:
        if "import sys\n" not in text:
            fail("cli.py import sys anchor missing")
        text = text.replace("import sys\n", "import sys\nimport json\n", 1)
    if "pirs_app = typer.Typer" not in text:
        fail("cli.py missing pirs_app")
    text = text.rstrip() + cli_block()
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
        if token not in text:
            fail(f"cli patch missing token: {token}")
    write(path, text)


def write_tests() -> None:
    unit = r'''
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
'''
    integration = r'''
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pegasus.cli import app


runner = CliRunner()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_slice16e_cli_inspects_planning_artifacts_read_only(tmp_path: Path) -> None:
    candidates = tmp_path / "pirs_field_candidates.json"
    selection = tmp_path / "pirs_selection_plan.json"
    design = tmp_path / "pirs_design_plan.json"
    readiness = tmp_path / "pirs_design_readiness.json"

    _write(candidates, {"gate": "pirs_candidate_gate", "candidate_count": 2, "rejected_count": 1, "candidates": [{}, {}], "rejected": [{}]})
    _write(selection, {"artifact": "pirs_selection_plan", "status": "planned", "budget": "fast", "selected_outcome_field_id": "field:outcome", "selected_covariate_field_ids": ["field:cov"], "selection_rejected": []})
    _write(design, {"artifact": "pirs_design_plan", "status": "planned", "design_matrix_state": "planned_only", "model_fit_state": "not_started", "residual_state": "not_started", "terms": [{}, {}]})
    _write(readiness, {"artifact": "pirs_design_readiness_gate", "status": "blocked", "ready": False, "accepted_fields": [], "rejected_fields": [{"field_id": "x"}]})

    result = runner.invoke(app, ["pirs", "inspect-candidates", "--manifest", str(candidates)])
    assert result.exit_code == 0, result.output
    assert '"candidate_count": 2' in result.output
    assert '"read_only": true' in result.output

    result = runner.invoke(app, ["pirs", "inspect-selection-plan", "--plan", str(selection)])
    assert result.exit_code == 0, result.output
    assert '"selected_covariate_count": 1' in result.output

    result = runner.invoke(app, ["pirs", "inspect-design-plan", "--plan", str(design)])
    assert result.exit_code == 0, result.output
    assert '"design_matrix_state": "planned_only"' in result.output

    result = runner.invoke(app, ["pirs", "inspect-design-readiness", "--manifest", str(readiness)])
    assert result.exit_code == 0, result.output
    assert '"ready": false' in result.output
    assert '"rejected_field_count": 1' in result.output


def test_slice16e_cli_help_lists_pirs_planning_commands() -> None:
    result = runner.invoke(app, ["pirs", "--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "candidates-from-run",
        "attach-candidate-gate",
        "plan-selection",
        "attach-selection-plan",
        "plan-design",
        "attach-design-plan",
        "check-design-readiness",
        "attach-design-readiness",
    ):
        assert command in result.output
'''
    write("tests/unit/test_slice16e_pirs_cli_boundary.py", unit)
    write("tests/integration/test_slice16e_cli_pirs_planning.py", integration)


def write_audit() -> None:
    audit = r'''
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
'''
    write("scripts/dev/audits/audit_slice16e_pirs_cli_boundary.py", audit)


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    cli = read("src/pegasus/cli.py")
    for token in ("candidates-from-run", "plan-selection", "plan-design", "check-design-readiness"):
        if token not in cli:
            fail(f"self-validate: missing CLI command token {token}")


def main() -> None:
    preflight()
    patch_cli()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 16E updater applied: PIRS planning CLI boundary exposed.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
