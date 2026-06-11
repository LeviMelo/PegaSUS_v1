from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _field_metadata(row):
    support = _loads(row.get("support_json"))
    if isinstance(support, dict) and isinstance(support.get("field_metadata"), dict):
        return support["field_metadata"]
    return _loads(row.get("metadata_json") or row.get("metadata"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    errors: list[str] = []

    v = pl.read_parquet(run / "V_fields.parquet").to_dicts()
    q = pl.read_parquet(run / "Q_tensor.parquet").to_dicts()
    warnings = pl.read_parquet(run / "Warnings.parquet").to_dicts()
    failed = pl.read_parquet(run / "FailedBranches.parquet").to_dicts()
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

    ids = {row["field_id"] for row in v}
    required = {
        "sidra_stitched_gdp_context",
        "sidra_projected_labor_context",
        "sidra_highdim_bounded_context",
        "sidra_stdfm_blocked_candidate",
    }
    missing = sorted(required - ids)
    if missing:
        errors.append(f"missing Slice 7A fields: {missing}")
    q_ids = {row["field_id"] for row in q}
    if not required.issubset(q_ids):
        errors.append("Q_tensor does not cover all Slice 7A fields")

    by_id = {row["field_id"]: row for row in v}
    if "sidra_projected_labor_context" in by_id:
        meta = _field_metadata(by_id["sidra_projected_labor_context"])
        if not meta or meta.get("projection", {}).get("fractional") is not True:
            errors.append("fractional projection metadata missing")
    if "sidra_highdim_bounded_context" in by_id:
        meta = _field_metadata(by_id["sidra_highdim_bounded_context"])
        if not meta or meta.get("high_dimensional_bound", {}).get("status") != "bounded":
            errors.append("high-dimensional bounded pushforward metadata missing")
    if "sidra_stdfm_blocked_candidate" in by_id:
        row = by_id["sidra_stdfm_blocked_candidate"]
        meta = _field_metadata(row)
        if row.get("state") != "blocked" or str(row.get("dashboard_safe")) != "False":
            errors.append("ST-DFM candidate is not blocked/non-dashboard-safe")
        if not meta or meta.get("stdfm_output", {}).get("status") != "blocked_solver_pending":
            errors.append("ST-DFM blocked_solver_pending metadata missing")

    codes = {row["code"] for row in warnings}
    for code in [
        "sidra_stitch_segment_provenance",
        "sidra_fractional_classification_projection",
        "high_dimensional_bounded_pushforward",
        "blocked_solver_pending",
    ]:
        if code not in codes:
            errors.append(f"missing warning code: {code}")
    if not any("blocked_solver_pending" in row.get("reason", "") for row in failed):
        errors.append("FailedBranches missing ST-DFM blocked_solver_pending branch")
    if not any("bounded pushforward" in row.get("reason", "") for row in failed):
        errors.append("FailedBranches missing unbounded high-dimensional SIDRA rejection")

    sidra_context = manifest.get("sidra_context", {})
    if sidra_context.get("stdfm_output", {}).get("status") != "blocked_solver_pending":
        errors.append("ReproducibilityManifest missing ST-DFM blocked metadata")
    for table in [
        "sidra_stitching_segments.parquet",
        "sidra_projection_matrix.parquet",
        "sidra_high_dimensional_bounds.parquet",
        "stdfm_certification.parquet",
        "stdfm_objective_contract.parquet",
    ]:
        if not (run / "Tables" / table).exists():
            errors.append(f"missing table: Tables/{table}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("AUDIT PASSED: Slice 7A SIDRA stitching, projection, high-dimensional bounding, and ST-DFM blocked certification contracts validated.")


if __name__ == "__main__":
    main()
