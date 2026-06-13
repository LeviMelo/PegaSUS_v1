
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(Path("src/pegasus/pirs/design_matrix.py"), "build_pirs_design_matrix_manifest") != 1:
        errors.append("design matrix builder API missing")
    if _count_defs(Path("src/pegasus/workflows/pirs_matrix.py"), "run_attach_pirs_design_matrix_to_run") != 1:
        errors.append("design matrix workflow API missing")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "planned", "outcome_field_id": "y", "covariate_field_ids": ["x"]})
        _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "ready", "ready": True})
        pq.write_table(pa.Table.from_pylist([
            {"field_id": "y", "values_json": "[1, 2]"},
            {"field_id": "x", "values_json": "[3, 4]"},
        ]), run_dir / "Q_tensor.parquet")
        result = run_attach_pirs_design_matrix_to_run(run_dir=run_dir)
        gate = result.get("pirs_design_matrix_gate", {})
        if gate.get("status") != "ready":
            errors.append("ready synthetic bundle did not produce ready design matrix gate")
        if not (run_dir / "Tables" / "pirs_design_matrix.parquet").exists():
            errors.append("ready synthetic bundle did not write pirs_design_matrix.parquet")
        p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
        if "pirs_design_matrix_gate" not in p_vector:
            errors.append("P_vector.json missing pirs_design_matrix_gate")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 17A PIRS design matrix materialization boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
