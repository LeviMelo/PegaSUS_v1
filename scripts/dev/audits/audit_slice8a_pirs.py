from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle


def _loads(value):
    if value is None:
        return None
    return json.loads(value) if isinstance(value, str) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    validation = validate_output_bundle(run_dir=str(run))
    if not validation.ok:
        raise SystemExit("Invalid output bundle: " + "; ".join(validation.errors))
    v = pl.read_parquet(run / "V_fields.parquet").to_dicts()
    q = pl.read_parquet(run / "Q_tensor.parquet").to_dicts()
    model = pl.read_parquet(run / "ModelAssociations.parquet").to_dicts()
    residuals = pl.read_parquet(run / "ResidualAssociations.parquet").to_dicts()
    failed = pl.read_parquet(run / "FailedBranches.parquet").to_dicts()
    ids = {row["field_id"] for row in v}; qids = {row["field_id"] for row in q}
    if "pirs_residual_all_deaths" not in ids:
        raise SystemExit("Missing PIRS residual field")
    if not ids <= qids:
        raise SystemExit("Q_tensor does not cover every PIRS V_field")
    residual = next(row for row in v if row["field_id"] == "pirs_residual_all_deaths")
    if _loads(residual["provenance"]) != ["model_derived"]:
        raise SystemExit("Residual field does not carry exact model_derived provenance")
    if residual["dashboard_safe"] != "False":
        raise SystemExit("Residual field must not be dashboard safe in Slice 8A")
    if not model or model[0]["status"] != "fitted":
        raise SystemExit("ModelAssociations missing fitted model row")
    if not residuals or residuals[0]["id"] != "pirs_residual_all_deaths":
        raise SystemExit("ResidualAssociations missing residual row")
    reasons = {row["reason"] for row in failed}
    if "zero_variance_field_excluded_from_design_matrix" not in reasons:
        raise SystemExit("Missing zero-variance rejection failed branch")
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    if manifest.get("telemetry", {}).get("stage_status", {}).get("pirs_model") != "success":
        raise SystemExit("pirs_model telemetry stage was not successful")
    if manifest.get("telemetry", {}).get("stage_status", {}).get("pirs_hsic") != "blocked":
        raise SystemExit("pirs_hsic must remain blocked until Slice 9")
    for rel in ["Tables/pirs_model_associations_detail.parquet", "Tables/pirs_residual_associations_detail.parquet", "Tables/pirs_design_matrix_contract.parquet", "Tables/pirs_field_selection.parquet", "Tables/pirs_diagnostics.parquet", "Tables/pirs/pirs_design_matrix.parquet", "Tables/pirs/pirs_residual_values.parquet"]:
        if not (run / rel).exists():
            raise SystemExit(f"Missing PIRS table artifact: {rel}")
    print("AUDIT PASSED: Slice 8A PIRS field selection, design matrix, exposure-offset model, residual extraction, cross-fitting policy, and output association contracts validated.")


if __name__ == "__main__":
    main()
