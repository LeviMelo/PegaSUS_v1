from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "14C"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/cli.py",
    "tests/unit/test_slice14c_efg_materialization_cli.py",
    "tests/integration/test_slice14c_cli_efg_materialization.py",
    "scripts/dev/audits/audit_slice14c_efg_materialization_cli.py",
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

MARKER_START = "# ---- Slice 14C EFG materialization CLI boundary ----"
MARKER_END = "# ---- End Slice 14C EFG materialization CLI boundary ----"


def fail(message: str) -> None:
    raise SystemExit(f"[slice14c] {message}")


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
        fail("CLI contains Slice 14C start marker without end marker")
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
        "src/pegasus/efg/materialize.py",
        "src/pegasus/efg/materialization_manifest.py",
        "src/pegasus/workflows/efg_materialize.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")

    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_defs(compile_text, "run_compile") != 1:
        fail("compile.py must have exactly one public run_compile before Slice 14C")
    if top_level_defs(compile_text, "_run_compile_impl") != 1:
        fail("compile.py must have exactly one private _run_compile_impl before Slice 14C")

    manifest_text = read("src/pegasus/efg/materialization_manifest.py")
    for token in (
        "build_efg_materialization_manifest",
        "write_efg_materialization_manifest",
        "attach_efg_materialization_summary_to_run",
        "efg_materialization_summary",
    ):
        if token not in manifest_text:
            fail(f"materialization_manifest.py missing Slice 14B API: {token}")

    workflow_text = read("src/pegasus/workflows/efg_materialize.py")
    for token in ("run_materialize_substrate_manifest", "run_attach_efg_materialization_to_run"):
        if token not in workflow_text:
            fail(f"efg_materialize workflow missing Slice 14B API: {token}")

    cli_text = read("src/pegasus/cli.py")
    if "efg_app = typer.Typer" not in cli_text or "app.add_typer(efg_app" not in cli_text:
        fail("cli.py does not expose efg_app Typer subcommand boundary")


def cli_block() -> str:
    return dedent(r'''

    # ---- Slice 14C EFG materialization CLI boundary ----
    @efg_app.command("materialize-substrate-manifest")
    def efg_materialize_substrate_manifest(
        substrate_manifest: Path = typer.Option(..., "--substrate-manifest"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        """Build a metadata-only EFG materialization manifest from a substrate manifest."""
        import json

        from pegasus.workflows.efg_materialize import run_materialize_substrate_manifest

        payload = run_materialize_substrate_manifest(substrate_manifest=substrate_manifest, output=output)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


    @efg_app.command("attach-materialization")
    def efg_attach_materialization_manifest(
        run_dir: Path = typer.Option(..., "--run-dir"),
        substrate_manifest: Path | None = typer.Option(None, "--substrate-manifest"),
    ) -> None:
        """Attach EFG substrate materialization metadata to an existing run bundle."""
        import json

        from pegasus.workflows.efg_materialize import run_attach_efg_materialization_to_run

        payload = run_attach_efg_materialization_to_run(run_dir=run_dir, substrate_manifest=substrate_manifest)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


    @efg_app.command("inspect-materialization")
    def efg_inspect_materialization_manifest(
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        """Inspect a metadata-only EFG materialization manifest without mutating a run."""
        import json

        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise typer.BadParameter(f"EFG materialization manifest is not a JSON object: {manifest}")
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = {
                "status": "evaluated",
                "materialization_id": payload.get("materialization_id"),
                "substrate_id": payload.get("substrate_id"),
                "field_count": payload.get("field_count"),
                "excluded_field_count": payload.get("excluded_field_count"),
                "metadata_only": payload.get("metadata_only"),
                "writes_v_fields": payload.get("writes_v_fields"),
                "writes_e_dag": payload.get("writes_e_dag"),
            }
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    # ---- End Slice 14C EFG materialization CLI boundary ----
    ''')


def patch_cli() -> None:
    path = "src/pegasus/cli.py"
    text = strip_marker_block(read(path))
    if "def efg_materialize_substrate_manifest" in text or "def efg_attach_materialization_manifest" in text:
        fail("cli.py contains unmarked Slice 14C EFG materialization command definitions")
    text = text.rstrip() + cli_block()
    for token in (
        '@efg_app.command("materialize-substrate-manifest")',
        '@efg_app.command("attach-materialization")',
        '@efg_app.command("inspect-materialization")',
        "def efg_materialize_substrate_manifest",
        "def efg_attach_materialization_manifest",
        "def efg_inspect_materialization_manifest",
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
        from pegasus.workflows.efg_materialize import (
            run_attach_efg_materialization_to_run,
            run_materialize_substrate_manifest,
        )

        assert callable(run_materialize_substrate_manifest)
        assert callable(run_attach_efg_materialization_to_run)
    ''')


def integration_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import polars as pl
    from typer.testing import CliRunner

    from pegasus.cli import app
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest


    def _build_run_with_substrate(tmp_path: Path) -> tuple[Path, Path]:
        run_dir = tmp_path / "run"
        artifact = tmp_path / "sim.parquet"
        create_empty_output_bundle(run_dir)
        pl.DataFrame(
            {
                "year": [2020, 2021, 2021],
                "age_years": [50, 51, 52],
                "race_color_admin": ["1", "4", ""],
                "underlying_icd_norm": ["I10", "J18", "R99"],
                "constant_col": [1, 1, 1],
                "all_missing_col": [None, None, None],
            }
        ).write_parquet(artifact)
        bundle = build_substrate_bundle(
            artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
        )
        substrate_manifest = run_dir / "Tables" / "substrate_manifest.json"
        write_substrate_bundle_manifest(bundle, substrate_manifest)
        return run_dir, substrate_manifest


    def test_slice14c_cli_materializes_inspects_and_attaches_without_new_first_class_key(tmp_path: Path) -> None:
        runner = CliRunner()
        run_dir, substrate_manifest = _build_run_with_substrate(tmp_path)
        output = tmp_path / "efg_materialization.json"

        materialize_result = runner.invoke(
            app,
            [
                "efg",
                "materialize-substrate-manifest",
                "--substrate-manifest",
                str(substrate_manifest),
                "--output",
                str(output),
            ],
        )
        assert materialize_result.exit_code == 0, materialize_result.output
        assert output.exists()
        materialized = json.loads(output.read_text(encoding="utf-8"))
        assert materialized["metadata_only"] is True
        assert materialized["writes_v_fields"] is False
        assert materialized["writes_e_dag"] is False
        assert any(item["field"]["unit"] == "ICD10" and item["field"]["kind"] == "observer_proxy" for item in materialized["fields"])

        inspect_result = runner.invoke(
            app,
            ["efg", "inspect-materialization", "--manifest", str(output)],
        )
        assert inspect_result.exit_code == 0, inspect_result.output
        assert "field_count" in inspect_result.output
        assert "writes_v_fields" in inspect_result.output

        before_keys = sorted(path.name for path in run_dir.iterdir())
        attach_result = runner.invoke(
            app,
            ["efg", "attach-materialization", "--run-dir", str(run_dir)],
        )
        after_keys = sorted(path.name for path in run_dir.iterdir())
        assert attach_result.exit_code == 0, attach_result.output
        assert before_keys == after_keys
        assert (run_dir / "Tables" / "efg_substrate_materialization.json").exists()
        assert validate_output_bundle(run_dir=str(run_dir)).ok

        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        assert run_config["efg_materialization_gate"]["status"] == "evaluated"
        assert run_config["efg_materialization_gate"]["writes_v_fields"] is False
    ''')


def audit_script() -> str:
    return dedent(r'''
    from __future__ import annotations

    import ast
    import json
    import tempfile
    from pathlib import Path

    import polars as pl
    from typer.testing import CliRunner


    def _count_defs(path: Path, name: str) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


    def main() -> int:
        errors: list[str] = []
        compile_path = Path("src/pegasus/workflows/compile.py")
        cli_path = Path("src/pegasus/cli.py")
        cli_text = cli_path.read_text(encoding="utf-8")

        if _count_defs(compile_path, "run_compile") != 1:
            errors.append("compile.py must contain exactly one public run_compile")
        if _count_defs(compile_path, "_run_compile_impl") != 1:
            errors.append("compile.py must contain exactly one private _run_compile_impl")
        for command in (
            '@efg_app.command("materialize-substrate-manifest")',
            '@efg_app.command("attach-materialization")',
            '@efg_app.command("inspect-materialization")',
        ):
            if cli_text.count(command) != 1:
                errors.append(f"CLI command declaration must occur exactly once: {command}")

        from pegasus.cli import app
        from pegasus.output.bundle import create_empty_output_bundle
        from pegasus.output.validate import validate_output_bundle
        from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest
        from pegasus.workflows.efg_materialize import run_attach_efg_materialization_to_run, run_materialize_substrate_manifest

        if not callable(run_materialize_substrate_manifest):
            errors.append("run_materialize_substrate_manifest is not callable")
        if not callable(run_attach_efg_materialization_to_run):
            errors.append("run_attach_efg_materialization_to_run is not callable")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            artifact = tmp_path / "sim.parquet"
            create_empty_output_bundle(run_dir)
            pl.DataFrame({
                "year": [2020, 2021, 2021],
                "age_years": [40, 41, 42],
                "underlying_icd_norm": ["I10", "J18", "R99"],
                "constant_col": [1, 1, 1],
                "all_missing_col": [None, None, None],
            }).write_parquet(artifact)
            bundle = build_substrate_bundle(
                artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
            )
            substrate_manifest = run_dir / "Tables" / "substrate_manifest.json"
            write_substrate_bundle_manifest(bundle, substrate_manifest)

            runner = CliRunner()
            output = tmp_path / "efg_materialization.json"
            result = runner.invoke(app, ["efg", "materialize-substrate-manifest", "--substrate-manifest", str(substrate_manifest), "--output", str(output)])
            if result.exit_code != 0:
                errors.append(f"CLI materialize-substrate-manifest failed: {result.output}")
            elif not output.exists():
                errors.append("CLI materialize-substrate-manifest did not write output")
            else:
                manifest = json.loads(output.read_text(encoding="utf-8"))
                if manifest.get("writes_v_fields") is not False or manifest.get("writes_e_dag") is not False:
                    errors.append("EFG CLI materialization must not write V_fields or E_DAG")

            before_keys = sorted(path.name for path in run_dir.iterdir())
            attach = runner.invoke(app, ["efg", "attach-materialization", "--run-dir", str(run_dir)])
            after_keys = sorted(path.name for path in run_dir.iterdir())
            if attach.exit_code != 0:
                errors.append(f"CLI attach-materialization failed: {attach.output}")
            if before_keys != after_keys:
                errors.append("CLI attach-materialization created a new first-class key")
            if not validate_output_bundle(run_dir=str(run_dir)).ok:
                errors.append("Output bundle validation failed after CLI attach-materialization")

        payload = {"ok": not errors, "errors": errors}
        print(json.dumps(payload, indent=2, sort_keys=True))
        if errors:
            return 1
        print("AUDIT PASSED: Slice 14C EFG materialization CLI boundary")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    ''')


def write_files() -> None:
    patch_cli()
    write("tests/unit/test_slice14c_efg_materialization_cli.py", unit_test())
    write("tests/integration/test_slice14c_cli_efg_materialization.py", integration_test())
    write("scripts/dev/audits/audit_slice14c_efg_materialization_cli.py", audit_script())


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import pegasus.cli as cli

    for name in (
        "efg_materialize_substrate_manifest",
        "efg_attach_materialization_manifest",
        "efg_inspect_materialization_manifest",
    ):
        if not hasattr(cli, name):
            fail(f"post-validate: pegasus.cli missing {name}")


def main() -> None:
    preflight()
    write_files()
    self_validate()
    print("Slice 14C updater applied: EFG materialization CLI boundary added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
