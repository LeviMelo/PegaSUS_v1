
from __future__ import annotations

import ast
import inspect
import json
import tempfile
from pathlib import Path

import polars as pl


def _top_level_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    materialize_path = Path("src/pegasus/efg/materialize.py")
    compile_path = Path("src/pegasus/workflows/compile.py")
    if "BlockedModuleError" in materialize_path.read_text(encoding="utf-8"):
        errors.append("efg/materialize.py still contains blocked scaffold")
    if _top_level_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must keep exactly one public run_compile")
    if _top_level_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must keep exactly one private _run_compile_impl")

    from pegasus.efg.materialize import materialize_substrate_bundle, materialized_field_nodes
    from pegasus.she.source_registry import resolve_source_fields
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle

    if "allow_heuristic" not in inspect.signature(resolve_source_fields).parameters:
        errors.append("resolve_source_fields lost allow_heuristic")

    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "sim.parquet"
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
        result = materialize_substrate_bundle(bundle)
        nodes = materialized_field_nodes(bundle)
        if result.field_count != len(bundle.candidates):
            errors.append("materialized field count does not match substrate candidates")
        if result.excluded_field_count != len(bundle.exclusions):
            errors.append("exclusion count does not match substrate exclusions")
        if {f.support["column"] for f in nodes} & {e.column for e in bundle.exclusions}:
            errors.append("excluded substrate fields were promoted to FieldNode")
        if not any(field.unit == "ICD10" for field in nodes):
            errors.append("diagnostic ICD10 observer field was not materialized")
        if not all(field.materialization_state == "metadata_only" for field in nodes):
            errors.append("Slice 14A must emit metadata_only FieldNodes only")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 14A EFG substrate materialization boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
