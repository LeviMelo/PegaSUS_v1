from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

FORBIDDEN_ARCHITECTURE_VALUES = {
    "compatibility_materializer",
    "pegasus.workflows.efg.run_build_sim_fixture",
}
FORBIDDEN_COMPILE_SYMBOLS = {
    "build_sim_fixture_efg_run",
    "build_sinasc_fixture_efg_run",
    "build_cnes_sih_fixture_bundle",
    "build_sidra_stdfm_fixture_bundle",
    "build_hsic_fixture_bundle",
    "build_population_tensor_fixture_bundle",
    "run_build_sim_fixture",
    "run_build_cnes_sih_fixture",
}
REQUIRED_ARCHITECTURE_VALUES = {
    "autonomous_efg_core",
    "autonomous_compiler_services",
    "quarantined_fixture_only",
}


def _literal_return_dict(function: ast.FunctionDef) -> dict:
    for node in ast.walk(function):
        if isinstance(node, ast.Return):
            return ast.literal_eval(node.value)
    raise AssertionError("function has no literal return")


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"{path} expected one {name}, found {len(matches)}")
    return matches[0]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    compile_source = compile_path.read_text(encoding="utf-8")
    compile_tree = ast.parse(compile_source)
    try:
        metadata = _literal_return_dict(_function(compile_path, "_compiler_architecture_metadata"))
    except Exception as exc:  # pragma: no cover - audit diagnostic path
        errors.append(f"cannot inspect _compiler_architecture_metadata: {exc}")
        metadata = {}
    serialized = json.dumps(metadata, sort_keys=True)
    for token in FORBIDDEN_ARCHITECTURE_VALUES:
        if token in serialized:
            errors.append(f"forbidden legacy architecture token still present: {token}")
    for token in REQUIRED_ARCHITECTURE_VALUES:
        if token not in serialized:
            errors.append(f"required autonomous architecture token missing: {token}")
    if metadata.get("legacy_graph_authority") is not False:
        errors.append("legacy_graph_authority must be False")
    if metadata.get("numerical_materialization") == "legacy_bootstrap":
        errors.append("numerical_materialization must not be legacy_bootstrap")
    if metadata.get("legacy_bootstrap_builder") not in {None, ""}:
        errors.append("legacy_bootstrap_builder must be None/empty in quarantined architecture")
    if metadata.get("legacy_bootstrap_status") != "quarantined_fixture_only":
        errors.append("legacy_bootstrap_status must be quarantined_fixture_only")
    imported_or_called = {
        node.id
        for node in ast.walk(compile_tree)
        if isinstance(node, ast.Name)
    }
    imported_or_called.update(
        alias.name
        for node in ast.walk(compile_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    )
    forbidden_runtime = sorted(FORBIDDEN_COMPILE_SYMBOLS.intersection(imported_or_called))
    if forbidden_runtime:
        errors.append(f"production compile references fixture runtime symbols: {forbidden_runtime}")
    if not Path("src/pegasus/workflows/stage_plan.py").exists():
        errors.append("stage_plan workflow module missing")
    contracts_text = Path("src/pegasus/acceptance/contracts.py").read_text(encoding="utf-8")
    for token in ("validate_compiler_stage_plan", "compiler_stage_plan_present", "stage_skip_proofs_valid"):
        if token not in contracts_text:
            errors.append(f"acceptance contract missing token: {token}")
    cli_text = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
    if '@acceptance_app.command("level3")' not in cli_text:
        errors.append("CLI missing pegasus acceptance level3 command")
    if args.run:
        run = Path(args.run)
        run_config = _load_json(run / "RunConfig.json")
        manifest = _load_json(run / "ReproducibilityManifest.json")
        if "compiler_stage_plan" not in run_config:
            errors.append("RunConfig.json missing compiler_stage_plan")
        if "compiler_stage_plan" not in manifest:
            errors.append("ReproducibilityManifest.json missing compiler_stage_plan")
        if run_config.get("compiler_architecture", {}).get("numerical_materialization") != "autonomous_compiler_services":
            errors.append("run compiler_architecture does not declare autonomous compiler services")
    payload = {"ok": not errors, "errors": errors, "architecture": metadata}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 26A/26B runtime authority quarantine and stage-plan proofs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
