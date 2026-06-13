
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.pirs_pipeline import run_pirs_planning_pipeline


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(Path("src/pegasus/workflows/pirs_pipeline.py"), "run_pirs_planning_pipeline") != 1:
        errors.append("pirs_pipeline workflow API missing")
    for path in (
        "src/pegasus/workflows/pirs_candidates.py",
        "src/pegasus/workflows/pirs_selection.py",
        "src/pegasus/workflows/pirs_design.py",
        "src/pegasus/workflows/pirs_readiness.py",
    ):
        if not Path(path).exists():
            errors.append(f"missing planning dependency: {path}")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        result = run_pirs_planning_pipeline(run_dir=run_dir, budget="fast")
        gate = result.get("pirs_planning_pipeline_gate", {})
        if gate.get("status") != "blocked":
            errors.append("empty run bundle should produce a blocked PIRS planning pipeline")
        for name in (
            "pirs_field_candidates.json",
            "pirs_selection_plan.json",
            "pirs_design_plan.json",
            "pirs_design_readiness.json",
            "pirs_planning_pipeline.json",
        ):
            if not (run_dir / "Tables" / name).exists():
                errors.append(f"pipeline did not write Tables/{name}")
        p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
        if "pirs_planning_pipeline_gate" not in p_vector:
            errors.append("P_vector.json missing pirs_planning_pipeline_gate")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16F PIRS planning pipeline orchestrator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
