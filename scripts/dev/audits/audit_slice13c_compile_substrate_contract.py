
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
    compile_path = Path("src/pegasus/workflows/compile.py")
    source_registry_path = Path("src/pegasus/she/source_registry.py")

    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must contain exactly one public top-level run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must contain exactly one private _run_compile_impl")
    if _count_defs(source_registry_path, "resolve_source_fields") != 1:
        errors.append("source_registry.py must contain exactly one top-level resolve_source_fields")

    from pegasus.she.source_registry import resolve_source_field, resolve_source_fields, source_registry_manifest
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle

    if "allow_heuristic" not in inspect.signature(resolve_source_fields).parameters:
        errors.append("resolve_source_fields missing allow_heuristic parameter")
    manifest = source_registry_manifest()
    if manifest.get("entry_count", 0) < 40:
        errors.append("source_registry_manifest missing top-level entry_count >= 40")
    race = resolve_source_field(source_system="SIM-DO", column_name="race_color_admin")
    if str(race.spec.carrier) != "Deaths":
        errors.append("race_color_admin carrier does not render as canonical Deaths")
    if not (race.spec.carrier == "deaths"):
        errors.append("CarrierId legacy equality with deaths failed")
    icd = resolve_source_field(source_system="SIM-DO", column_name="underlying_icd_norm")
    if icd.spec.unit != "ICD10" or icd.spec.aggregation != "non_aggregable":
        errors.append("underlying_icd_norm did not resolve as ICD10/non_aggregable")
    ref = SourceArtifactRef(path="missing.parquet", source_system="SIM-DO", provenance_mode="fixture")
    if ref.artifact_role != "processed_events":
        errors.append("SourceArtifactRef default artifact_role changed")
    bundle = build_substrate_bundle(artifacts=[ref])
    if bundle.summary().get("excluded_field_count", 0) < 1:
        errors.append("build_substrate_bundle did not quarantine missing artifact")

    payload = {"ok": not errors, "errors": errors, "manifest_entry_count": manifest.get("entry_count")}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 13C compile/substrate contract consolidated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
