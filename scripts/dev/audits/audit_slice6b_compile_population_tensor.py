from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Slice 6B compile-integrated Population Tensor contract.")
    parser.add_argument("--run", required=True, help="Compile run directory emitted from alagoas_smoke_population_tensor.json")
    args = parser.parse_args()

    run_dir = Path(args.run)
    errors: list[str] = []

    validation = validate_output_bundle(run_dir=str(run_dir))
    if not validation.ok:
        errors.extend(validation.errors)

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    warnings = pl.read_parquet(run_dir / "Warnings.parquet")
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

    ids = set(v["field_id"].to_list())
    if "population_tensor_independent_denominator" not in ids:
        errors.append("population_tensor_independent_denominator field missing from V_fields")

    q_ids = set(q["field_id"].to_list())
    if "population_tensor_independent_denominator" not in q_ids:
        errors.append("population_tensor_independent_denominator missing from Q_tensor")

    meta = run_config.get("population_tensor")
    if not isinstance(meta, dict):
        errors.append("RunConfig.population_tensor missing")
        meta = {}
    if meta.get("mode") != "independent_denominator":
        errors.append(f"unexpected population tensor mode: {meta.get('mode')!r}")
    if meta.get("independent_denominator_mode") is not True:
        errors.append("independent_denominator_mode must be true for independent tensor compile run")
    if meta.get("attach_stage") != "population_solver":
        errors.append("population tensor attach_stage must be population_solver")

    manifest_meta = manifest.get("population_tensor")
    if not isinstance(manifest_meta, dict):
        errors.append("ReproducibilityManifest.population_tensor missing")
    elif manifest_meta.get("field_id") != meta.get("field_id"):
        errors.append("RunConfig/ReproducibilityManifest population_tensor field_id mismatch")

    if manifest.get("telemetry", {}).get("stage_status", {}).get("population_solver") != "success":
        errors.append("telemetry.population_solver must be success")

    if not (run_dir / "Tables" / "population_tensor_diagnostics.parquet").exists():
        errors.append("population tensor diagnostics table missing")

    warning_codes = set(warnings["code"].to_list()) if warnings.height and "code" in warnings.columns else set()
    if "independent_population_denominator_mode" not in warning_codes:
        errors.append("independent population denominator warning code missing")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("AUDIT PASSED: Slice 6B compile-integrated Population Tensor metadata, independent denominator mode, telemetry, Q coverage, and validator enforcement validated.")


if __name__ == "__main__":
    main()
