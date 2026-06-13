
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
from pegasus.workflows.hsic_execute import run_execute_hsic_residual_scan


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
    if _count_defs(Path("src/pegasus/pirs/hsic_run.py"), "build_hsic_residual_scan_manifest") != 1:
        errors.append("HSIC run builder API missing")
    cli = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
    if '@pirs_app.command("scan-residuals-hsic")' not in cli:
        errors.append("CLI scan-residuals-hsic command missing")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        matrix = run_dir / "Tables" / "pirs_design_matrix.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"row_id": 0, "intercept": 1.0, "response": 1.0, "covariate_001": 0.0},
            {"row_id": 1, "intercept": 1.0, "response": 2.5, "covariate_001": 1.0},
            {"row_id": 2, "intercept": 1.0, "response": 2.7, "covariate_001": 2.0},
            {"row_id": 3, "intercept": 1.0, "response": 4.8, "covariate_001": 3.0},
        ]), matrix)
        _write_json(run_dir / "Tables" / "pirs_design_matrix_manifest.json", {"status": "ready", "matrix_written": True, "matrix_path": str(matrix), "family": "gaussian_identity", "field_specs": [{"field_id": "outcome", "role": "outcome", "column": "response"}, {"field_id": "cov", "role": "covariate", "column": "covariate_001"}]})
        run_execute_pirs_model(run_dir=run_dir)
        result = run_execute_hsic_residual_scan(run_dir=run_dir)
        gate = result.get("hsic_residual_scan_gate", {})
        if gate.get("status") != "scanned":
            errors.append("synthetic run was not scanned")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.append("output bundle invalid after HSIC scan: " + "; ".join(validation.errors))
        for rel in ("Tables/hsic_residual_scan_manifest.json", "Tables/hsic_residual_scan_scores.parquet"):
            if not (run_dir / rel).exists():
                errors.append(f"missing HSIC artifact: {rel}")
        hypotheses = pq.read_table(run_dir / "Hypotheses.parquet").to_pylist()
        if not hypotheses:
            errors.append("HSIC hypothesis rows missing")
        else:
            row = hypotheses[0]
            required = ("hypothesis_id", "covariate_field_id", "residual_field_id", "hsic_mode", "fdr_method", "n_eff", "approximation_diagnostics_json")
            missing = [name for name in required if row.get(name) in (None, "")]
            if missing:
                errors.append("HSIC hypothesis row missing canonical schema values: " + ", ".join(missing))
            if row.get("hsic_mode") not in {"exact_linear", "disabled"}:
                errors.append("HSIC hypothesis row has unexpected hsic_mode")
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 18A HSIC residual scan integration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
