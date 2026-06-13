
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must still contain exactly one _run_compile_impl")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        before_keys = sorted(OUTPUT_BUNDLE_FILES.keys())
        field = {
            "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
            "name": "SIM-DO underlying ICD observer",
            "kind": "observer_proxy",
            "carrier": "Deaths",
            "unit": "ICD10",
            "aggregation": "non_aggregable",
            "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
            "source": ["SIM-DO"],
            "support": {"column": "underlying_icd_norm"},
            "axes": {"diagnostic": "ICD10"},
            "provenance": ["fixture"],
            "warnings": [],
            "lineage_hash": "abc",
            "registry_hash": "registry",
            "materialization_state": "metadata_only",
        }
        plan_path = run_dir / "Tables" / "efg_promotion_plan.json"
        plan_path.write_text(json.dumps({"promotions": [{"status": "planned", "action": "promote", "field": field}]}), encoding="utf-8")
        result = apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
        if result.promoted_field_count != 1:
            errors.append("expected exactly one promoted field")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.extend(validation.errors)
        v_ids = {row["field_id"] for row in pq.read_table(run_dir / "V_fields.parquet").to_pylist()}
        q_ids = {row["field_id"] for row in pq.read_table(run_dir / "Q_tensor.parquet").to_pylist()}
        vd_ids = {row["field_id"] for row in pq.read_table(run_dir / "VariableDictionary.parquet").to_pylist()}
        if field["field_id"] not in v_ids or field["field_id"] not in q_ids or field["field_id"] not in vd_ids:
            errors.append("promoted field missing from V_fields/Q_tensor/VariableDictionary")
        after_keys = sorted(OUTPUT_BUNDLE_FILES.keys())
        if before_keys != after_keys:
            errors.append("output bundle first-class key registry changed")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 15B EFG promotion apply boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
