
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import polars as pl


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must contain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must contain exactly one private _run_compile_impl")

    from pegasus.efg.materialization_manifest import attach_efg_materialization_summary_to_run
    from pegasus.efg.materialize import materialized_field_nodes
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest

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
        if not any(field.unit == "ICD10" and field.kind == "observer_proxy" for field in materialized_field_nodes(bundle)):
            errors.append("Slice 14A diagnostic observer_proxy materialization is unavailable")
        write_substrate_bundle_manifest(bundle, run_dir / "Tables" / "substrate_manifest.json")
        before_keys = sorted(path.name for path in run_dir.iterdir())
        summary = attach_efg_materialization_summary_to_run(run_dir=run_dir)
        after_keys = sorted(path.name for path in run_dir.iterdir())
        if before_keys != after_keys:
            errors.append("EFG materialization attach created a new first-class run key")
        if summary.get("writes_v_fields") is not False or summary.get("writes_e_dag") is not False:
            errors.append("Slice 14B must not write V_fields or E_DAG")
        if summary.get("field_count") != len(bundle.candidates):
            errors.append("EFG materialization field_count mismatch")
        if summary.get("excluded_field_count") != len(bundle.exclusions):
            errors.append("EFG materialization excluded_field_count mismatch")
        manifest_path = run_dir / "Tables" / "efg_substrate_materialization.json"
        if not manifest_path.exists():
            errors.append("EFG materialization manifest was not written under Tables")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            materialized_columns = {item["field"]["support"]["column"] for item in manifest.get("fields", [])}
            excluded_columns = {item.get("column") for item in manifest.get("excluded_source_fields", [])}
            if materialized_columns & excluded_columns:
                errors.append("Excluded substrate fields were promoted in EFG materialization manifest")
        if not validate_output_bundle(run_dir=str(run_dir)).ok:
            errors.append("Output bundle validation failed after EFG materialization attach")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 14B EFG materialization manifest attach")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
