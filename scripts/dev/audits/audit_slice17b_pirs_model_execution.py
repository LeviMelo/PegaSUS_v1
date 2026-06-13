
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.pirs_execute import run_execute_pirs_model


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py public run_compile count changed")
    if _count_defs(Path("src/pegasus/pirs/model_execution.py"), "build_pirs_model_execution_manifest") != 1:
        errors.append("model execution builder API missing")
    cli = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
    if '@pirs_app.command("execute-model")' not in cli:
        errors.append("CLI execute-model command missing")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        matrix = run_dir / "Tables" / "pirs_design_matrix.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"row_id": 0, "intercept": 1.0, "response": 1.0, "covariate_001": 0.0},
            {"row_id": 1, "intercept": 1.0, "response": 2.0, "covariate_001": 1.0},
            {"row_id": 2, "intercept": 1.0, "response": 3.0, "covariate_001": 2.0},
        ]), matrix)
        _write_json(run_dir / "Tables" / "pirs_design_matrix_manifest.json", {"status": "ready", "matrix_written": True, "matrix_path": str(matrix), "family": "gaussian_identity", "field_specs": [{"field_id": "outcome", "role": "outcome", "column": "response"}, {"field_id": "cov", "role": "covariate", "column": "covariate_001"}]})
        result = run_execute_pirs_model(run_dir=run_dir)
        gate = result.get("pirs_model_execution_gate", {})
        if gate.get("status") != "fitted":
            errors.append("synthetic run did not fit")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.append("output bundle invalid after PIRS model execution: " + "; ".join(validation.errors))
        for rel in ("Tables/pirs_model_execution_manifest.json", "Tables/pirs_model_coefficients.parquet", "Tables/pirs_fitted_values.parquet", "Tables/pirs_residual_values.parquet"):
            if not (run_dir / rel).exists():
                errors.append(f"missing model execution artifact: {rel}")
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 17B PIRS model execution and residual materialization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
