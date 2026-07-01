
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
    from pegasus.workflows.construct.efg_materialize import run_attach_efg_materialization_to_run, run_materialize_substrate_manifest

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
