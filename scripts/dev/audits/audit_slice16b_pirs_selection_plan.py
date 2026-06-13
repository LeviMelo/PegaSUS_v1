
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.pirs.selection_plan import attach_pirs_selection_plan_to_run, build_pirs_selection_plan


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _candidate(field_id: str, role: str, utility: float) -> dict:
    return {
        "field_id": field_id,
        "role": role,
        "utility": utility,
        "q_state": "verified",
        "carrier": "Deaths",
        "unit": "counts",
        "support": {"years": [2020]},
        "variance": 0.2,
        "warnings": [],
        "provenance": ["fixture"],
    }


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(Path("src/pegasus/pirs/selection_plan.py"), "build_pirs_selection_plan") != 1:
        errors.append("PIRS selection plan API missing")
    if _count_defs(Path("src/pegasus/pirs/run_candidates.py"), "build_pirs_candidates_from_run") != 1:
        errors.append("PIRS candidate gate API missing")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        candidate_manifest = run_dir / "Tables" / "pirs_field_candidates.json"
        candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        candidate_manifest.write_text(json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 2,
            "rejected_count": 1,
            "candidates": [_candidate("audit_outcome", "outcome", 9.0), _candidate("audit_covariate", "covariate", 4.0)],
            "rejected": [{"field_id": "metadata_only", "reason": "metadata_only_field_not_model_eligible"}],
        }), encoding="utf-8")
        plan = build_pirs_selection_plan(candidate_manifest=candidate_manifest, budget="standard")
        if plan.status != "planned":
            errors.append("selection plan did not reach planned status")
        if plan.selected_outcome_field_id != "audit_outcome":
            errors.append("selection plan did not select expected outcome")
        if plan.residual_mode != "cross_fitted":
            errors.append("standard budget did not select cross_fitted residual mode")
        summary = attach_pirs_selection_plan_to_run(run_dir=run_dir, candidate_manifest=candidate_manifest, budget="standard")
        if summary.get("model_fitted") is not False or summary.get("design_matrix_materialized") is not False:
            errors.append("selection gate summary must remain non-mutating/model-free")
        p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
        if p_vector.get("pirs_selection_gate", {}).get("selected_outcome_field_id") != "audit_outcome":
            errors.append("pirs_selection_gate summary was not attached to P_vector")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16B PIRS selection plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
