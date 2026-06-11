from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl


def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    errors: list[str] = []
    v = pl.read_parquet(run / "V_fields.parquet").to_dicts()
    ids = {row["field_id"] for row in v}
    required = {"hsic_outcome_residual_deviance", "hsic_covariate_context", "hsic_residual_association_score"}
    missing = sorted(required - ids)
    if missing:
        errors.append(f"missing HSIC fields: {missing}")
    by_id = {row["field_id"]: row for row in v}
    if "hsic_outcome_residual_deviance" in by_id:
        if _loads(by_id["hsic_outcome_residual_deviance"].get("provenance")) != ["model_derived"]:
            errors.append("residual field does not preserve model_derived provenance")
    if "hsic_residual_association_score" in by_id:
        row = by_id["hsic_residual_association_score"]
        if row.get("dashboard_safe") != "False":
            errors.append("HSIC score must not be dashboard-safe by default")
        support = _loads(row.get("support_json")) or {}
        metadata = support.get("field_metadata") or {}
        if metadata.get("residual_mode") not in {"cross_fitted", "cross_fitted_parametric_bootstrap", "in_sample"}:
            errors.append("HSIC residual mode metadata missing")
        if metadata.get("hsic_mode") not in {"exact", "nystrom", "rff", "disabled", "cuda_unavailable_abort"}:
            errors.append("HSIC mode metadata missing")
    q_ids = {row["field_id"] for row in pl.read_parquet(run / "Q_tensor.parquet").to_dicts()}
    if not required.issubset(q_ids):
        errors.append("Q_tensor does not cover all HSIC fields")
    hyp = pl.read_parquet(run / "Hypotheses.parquet").to_dicts()
    if not hyp:
        errors.append("Hypotheses table is empty")
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text())
    meta = manifest.get("pirs_hsic") or {}
    if not meta.get("approximation_diagnostics_emitted"):
        errors.append("approximation diagnostics flag missing")
    if meta.get("budget") in {"standard", "deep"} and not meta.get("standard_deep_cross_fitted_residuals"):
        errors.append("standard/deep HSIC did not certify cross-fitted residual use")
    for rel in [
        "Tables/hsic_outputs.parquet",
        "Tables/hsic_approximation_diagnostics.parquet",
        "Tables/hsic_null_regime.parquet",
        "Tables/hsic_fdr_correction.parquet",
    ]:
        if not (run / rel).exists():
            errors.append(f"missing table artifact: {rel}")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("AUDIT PASSED: Slice 9A HSIC residual scanner, null strategy, FDR correction, approximation diagnostics, and CUDA abort surface validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
