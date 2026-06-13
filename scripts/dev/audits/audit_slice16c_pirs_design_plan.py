
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.pirs.design_plan import build_pirs_design_plan
from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    design_path = Path("src/pegasus/pirs/design_plan.py")
    design_text = design_path.read_text(encoding="utf-8")

    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must retain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must retain exactly one private _run_compile_impl")
    for forbidden in ("fit_parametric_model", "ModelAssociations", "ResidualAssociations", "Hypotheses", "run_hsic"):
        if forbidden in design_text:
            errors.append(f"design_plan.py must not invoke or mention {forbidden}")

    plan = build_pirs_design_plan(
        {
            "selected_outcome_field_id": "field:outcome",
            "selected_covariate_field_ids": ["field:cov"],
            "residual_mode": "cross_fitted",
            "fold_scheme": {"mode": "cross_fitted", "fold_count": 5},
        }
    ).as_manifest()
    if plan.get("design_matrix_state") != "planned_only":
        errors.append("design plan must remain planned_only")
    if plan.get("model_fit_state") != "not_started":
        errors.append("design plan must not start model fitting")
    if plan.get("residual_state") != "not_started":
        errors.append("design plan must not create residuals")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        selection = run_dir / "Tables" / "pirs_selection_plan.json"
        selection.write_text(
            json.dumps(
                {
                    "selected_outcome_field_id": "field:outcome",
                    "selected_covariate_field_ids": ["field:cov"],
                    "residual_mode": "in_sample",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        attached = run_attach_pirs_design_plan_to_run(run_dir=run_dir, selection_plan=selection)
        if attached.get("status") != "planned":
            errors.append("attached design plan did not reach planned status")
        if not (run_dir / "Tables" / "pirs_design_plan.json").exists():
            errors.append("pirs_design_plan.json was not written")
        if not validate_output_bundle(run_dir=str(run_dir)).ok:
            errors.append("output bundle validation failed after design gate attach")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16C PIRS design plan boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
