
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.efg_promotion import run_attach_efg_promotion_plan_to_run


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py must contain exactly one public run_compile")
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "_run_compile_impl") != 1:
        errors.append("compile.py must contain exactly one _run_compile_impl")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        materialization_path = run_dir / "Tables" / "efg_substrate_materialization.json"
        materialization_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "materialized_fields": [
                        {
                            "field_id": "efg_substrate::underlying_icd_norm",
                            "name": "SIM-DO.underlying_icd_norm",
                            "kind": "observer_proxy",
                            "carrier": "Deaths",
                            "unit": "ICD10",
                            "dashboard_safe": "warning",
                            "materialization_state": "metadata_only",
                            "support": {"column": "underlying_icd_norm"},
                        }
                    ],
                    "excluded_source_fields": [
                        {"column": "constant_col", "reason": "zero_variance", "efg_materialized": False}
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        before = {p.name for p in run_dir.iterdir()}
        summary = run_attach_efg_promotion_plan_to_run(run_dir=run_dir)
        after = {p.name for p in run_dir.iterdir()}
        if before != after:
            errors.append("EFG promotion attach changed first-class run keys")
        if summary.get("writes_v_fields") is not False:
            errors.append("EFG promotion summary must state writes_v_fields=False")
        if summary.get("planned_count") != 1:
            errors.append("EFG promotion planned_count did not equal 1")
        if not (run_dir / "Tables" / "efg_promotion_plan.json").exists():
            errors.append("EFG promotion plan was not written under Tables")
        if not validate_output_bundle(run_dir=str(run_dir)).ok:
            errors.append("EFG promotion attach invalidated output bundle")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 15A EFG promotion plan boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
