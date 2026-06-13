from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "15D"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/cli.py",
    "tests/unit/test_slice15d_efg_promotion_cli.py",
    "tests/integration/test_slice15d_cli_efg_promotion.py",
    "scripts/dev/audits/audit_slice15d_efg_promotion_cli.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/source_registry.py",
    "src/pegasus/she/substrate.py",
    "src/pegasus/output/",
    "src/pegasus/pirs/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)

MARKER_START = "# ---- Slice 15D EFG promotion CLI boundary ----"
MARKER_END = "# ---- End Slice 15D EFG promotion CLI boundary ----"


def fail(message: str) -> None:
    raise SystemExit(f"[slice15d] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_defs(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def strip_marker_block(text: str) -> str:
    start = text.find(MARKER_START)
    if start == -1:
        return text.rstrip() + "\n"
    end = text.find(MARKER_END, start)
    if end == -1:
        fail("CLI contains Slice 15D start marker without end marker")
    end_line = text.find("\n", end)
    if end_line == -1:
        end_line = len(text)
    else:
        end_line += 1
    return (text[:start].rstrip() + "\n" + text[end_line:].lstrip()).rstrip() + "\n"


def preflight() -> None:
    required = (
        "src/pegasus/cli.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/workflows/efg_promotion.py",
        "src/pegasus/workflows/efg_apply.py",
        "src/pegasus/efg/promotion_plan.py",
        "src/pegasus/efg/promotion_apply.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")

    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_defs(compile_text, "run_compile") != 1:
        fail("compile.py must have exactly one public run_compile before Slice 15D")
    if top_level_defs(compile_text, "_run_compile_impl") != 1:
        fail("compile.py must have exactly one private _run_compile_impl before Slice 15D")

    cli_text = read("src/pegasus/cli.py")
    if "efg_app = typer.Typer" not in cli_text or "app.add_typer(efg_app" not in cli_text:
        fail("cli.py does not expose efg_app Typer subcommand boundary")
    for token in ("plan-promotion", "attach-promotion-plan", "apply-promotion-plan", "inspect-promotion-plan"):
        if token in cli_text and MARKER_START not in cli_text:
            fail(f"cli.py already contains unmarked Slice 15D token: {token}")

    promotion_workflow = read("src/pegasus/workflows/efg_promotion.py")
    for token in ("run_plan_efg_promotion", "run_attach_efg_promotion_plan_to_run"):
        if token not in promotion_workflow:
            fail(f"efg_promotion workflow missing Slice 15A API: {token}")

    apply_workflow = read("src/pegasus/workflows/efg_apply.py")
    if "run_apply_efg_promotion_plan" not in apply_workflow:
        fail("efg_apply workflow missing Slice 15B API: run_apply_efg_promotion_plan")


def cli_block() -> str:
    return dedent(r'''

    # ---- Slice 15D EFG promotion CLI boundary ----
    @efg_app.command("plan-promotion")
    def efg_plan_promotion(
        run_dir: Path = typer.Option(..., "--run-dir"),
        materialization_manifest: Path = typer.Option(..., "--materialization-manifest"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        """Build a non-mutating EFG promotion plan from a materialization manifest."""
        import json

        from pegasus.workflows.efg_promotion import run_plan_efg_promotion

        payload = run_plan_efg_promotion(
            run_dir=run_dir,
            materialization_manifest=materialization_manifest,
            output=output,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


    @efg_app.command("attach-promotion-plan")
    def efg_attach_promotion_plan(
        run_dir: Path = typer.Option(..., "--run-dir"),
        materialization_manifest: Path | None = typer.Option(None, "--materialization-manifest"),
    ) -> None:
        """Attach a non-mutating EFG promotion plan and gate summary to a run bundle."""
        import json

        from pegasus.workflows.efg_promotion import run_attach_efg_promotion_plan_to_run

        payload = run_attach_efg_promotion_plan_to_run(
            run_dir=run_dir,
            materialization_manifest=materialization_manifest,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


    @efg_app.command("apply-promotion-plan")
    def efg_apply_promotion_plan(
        run_dir: Path = typer.Option(..., "--run-dir"),
        promotion_plan: Path | None = typer.Option(None, "--promotion-plan"),
        validate: bool = typer.Option(True, "--validate/--no-validate"),
    ) -> None:
        """Apply planned EFG promotions to descriptive/quarantined bundle surfaces."""
        import json

        from pegasus.workflows.efg_apply import run_apply_efg_promotion_plan

        payload = run_apply_efg_promotion_plan(
            run_dir=run_dir,
            promotion_plan=promotion_plan,
            validate=validate,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


    @efg_app.command("inspect-promotion-plan")
    def efg_inspect_promotion_plan(
        plan: Path = typer.Option(..., "--plan"),
    ) -> None:
        """Inspect an EFG promotion plan without mutating a run bundle."""
        import json

        payload = json.loads(plan.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise typer.BadParameter(f"EFG promotion plan is not a JSON object: {plan}")
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = {
                "status": payload.get("status", "evaluated"),
                "promotion_plan_id": payload.get("promotion_plan_id"),
                "planned_promotion_count": payload.get("planned_promotion_count"),
                "conflict_count": payload.get("conflict_count"),
                "excluded_source_field_count": payload.get("excluded_source_field_count"),
                "mutates_v_fields": payload.get("mutates_v_fields", False),
            }
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    # ---- End Slice 15D EFG promotion CLI boundary ----
    ''')


def patch_cli() -> None:
    path = "src/pegasus/cli.py"
    text = strip_marker_block(read(path))
    for name in (
        "efg_plan_promotion",
        "efg_attach_promotion_plan",
        "efg_apply_promotion_plan",
        "efg_inspect_promotion_plan",
    ):
        if f"def {name}" in text:
            fail(f"cli.py contains unmarked Slice 15D command definition: {name}")
    text = text.rstrip() + cli_block()
    for token in (
        '@efg_app.command("plan-promotion")',
        '@efg_app.command("attach-promotion-plan")',
        '@efg_app.command("apply-promotion-plan")',
        '@efg_app.command("inspect-promotion-plan")',
        "def efg_plan_promotion",
        "def efg_attach_promotion_plan",
        "def efg_apply_promotion_plan",
        "def efg_inspect_promotion_plan",
    ):
        if token not in text:
            fail(f"CLI patch did not install token: {token}")
    write(path, text)


def unit_test() -> str:
    return dedent(r'''
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
    ''')


def integration_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    from typer.testing import CliRunner

    from pegasus.cli import app
    import pegasus.workflows.efg_apply as apply_workflow
    import pegasus.workflows.efg_promotion as promotion_workflow


    runner = CliRunner()


    def test_slice15d_plan_promotion_cli_delegates_to_workflow(tmp_path: Path, monkeypatch) -> None:
        run_dir = tmp_path / "run"
        materialization = tmp_path / "efg_materialization.json"
        materialization.write_text(json.dumps({"summary": {"field_count": 1}}), encoding="utf-8")

        def fake_plan(*, run_dir, materialization_manifest, output=None):
            assert Path(run_dir).name == "run"
            assert Path(materialization_manifest) == materialization
            assert output is None
            return {"summary": {"status": "evaluated", "planned_promotion_count": 1}}

        monkeypatch.setattr(promotion_workflow, "run_plan_efg_promotion", fake_plan)
        result = runner.invoke(
            app,
            [
                "efg",
                "plan-promotion",
                "--run-dir",
                str(run_dir),
                "--materialization-manifest",
                str(materialization),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["summary"]["planned_promotion_count"] == 1


    def test_slice15d_attach_promotion_plan_cli_delegates_to_workflow(tmp_path: Path, monkeypatch) -> None:
        run_dir = tmp_path / "run"

        def fake_attach(*, run_dir, materialization_manifest=None):
            assert Path(run_dir).name == "run"
            assert materialization_manifest is None
            return {"status": "evaluated", "planned_promotion_count": 2}

        monkeypatch.setattr(promotion_workflow, "run_attach_efg_promotion_plan_to_run", fake_attach)
        result = runner.invoke(app, ["efg", "attach-promotion-plan", "--run-dir", str(run_dir)])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["planned_promotion_count"] == 2


    def test_slice15d_apply_promotion_plan_cli_delegates_to_workflow(tmp_path: Path, monkeypatch) -> None:
        run_dir = tmp_path / "run"
        plan = tmp_path / "efg_promotion_plan.json"
        plan.write_text(json.dumps({"summary": {"planned_promotion_count": 1}}), encoding="utf-8")

        def fake_apply(*, run_dir, promotion_plan=None, validate=True):
            assert Path(run_dir).name == "run"
            assert Path(promotion_plan) == plan
            assert validate is False
            return {"gate": "efg_promotion_apply", "promoted_field_count": 1}

        monkeypatch.setattr(apply_workflow, "run_apply_efg_promotion_plan", fake_apply)
        result = runner.invoke(
            app,
            [
                "efg",
                "apply-promotion-plan",
                "--run-dir",
                str(run_dir),
                "--promotion-plan",
                str(plan),
                "--no-validate",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["promoted_field_count"] == 1


    def test_slice15d_inspect_promotion_plan_cli_is_read_only(tmp_path: Path) -> None:
        plan = tmp_path / "efg_promotion_plan.json"
        plan.write_text(
            json.dumps({"summary": {"status": "evaluated", "planned_promotion_count": 3}}),
            encoding="utf-8",
        )
        before = plan.read_text(encoding="utf-8")
        result = runner.invoke(app, ["efg", "inspect-promotion-plan", "--plan", str(plan)])
        after = plan.read_text(encoding="utf-8")
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["planned_promotion_count"] == 3
        assert after == before
    ''')


def audit_script() -> str:
    return dedent(r'''
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
        from pegasus.workflows.efg_promotion import run_attach_efg_promotion_plan_to_run, run_plan_efg_promotion
        from pegasus.workflows.efg_apply import run_apply_efg_promotion_plan

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
    ''')


def write_tests_and_audit() -> None:
    write("tests/unit/test_slice15d_efg_promotion_cli.py", unit_test())
    write("tests/integration/test_slice15d_cli_efg_promotion.py", integration_test())
    write("scripts/dev/audits/audit_slice15d_efg_promotion_cli.py", audit_script())


def self_validate() -> None:
    for path in TOUCH_LIST:
        if path.endswith(".py"):
            py_compile.compile(str(ROOT / path), doraise=True)
    sys.path.insert(0, str(ROOT / "src"))
    from pegasus import cli  # noqa: F401
    cli_text = read("src/pegasus/cli.py")
    for token in (
        '@efg_app.command("plan-promotion")',
        '@efg_app.command("attach-promotion-plan")',
        '@efg_app.command("apply-promotion-plan")',
        '@efg_app.command("inspect-promotion-plan")',
    ):
        if cli_text.count(token) != 1:
            fail(f"post-validate CLI token count not one: {token}")


def main() -> None:
    preflight()
    patch_cli()
    write_tests_and_audit()
    self_validate()
    print("Slice 15D updater applied: EFG promotion CLI boundary exposed.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
