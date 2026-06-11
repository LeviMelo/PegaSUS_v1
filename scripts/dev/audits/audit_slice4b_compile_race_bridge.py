from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle


def fail(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    failures: list[dict] = []
    validation = validate_output_bundle(run_dir=str(run))
    if not validation.ok:
        failures.append({"kind": "validate_output_bundle", "errors": validation.errors})
    v = pl.read_parquet(run / "V_fields.parquet")
    required_fields = {
        "SIMRaceAdminRawCount_1",
        "SIMRaceAdminRawCount_2",
        "SIMRaceAdminRawCount_3",
        "SIMRaceAdminRawCount_4",
        "SIMRaceAdminRawCount_5",
        "SIMRaceBridgeMissingRaceObserver",
        "SIMRaceBridgePosteriorCount_branca",
        "SIMRaceBridgePosteriorCount_preta",
        "SIMRaceBridgePosteriorCount_amarela",
        "SIMRaceBridgePosteriorCount_parda",
        "SIMRaceBridgePosteriorCount_indigena",
    }
    field_ids = set(v["field_id"].to_list())
    missing = sorted(required_fields - field_ids)
    if missing:
        failures.append({"kind": "missing_compile_race_bridge_fields", "missing": missing})
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
    for holder, name in [(manifest, "manifest"), (run_config, "run_config")]:
        bridge = holder.get("race_bridge")
        if not isinstance(bridge, dict):
            failures.append({"kind": f"missing_{name}_race_bridge"})
            continue
        for key in ["bridge_id", "mode", "prior_hash", "missing_race_share", "sensitivity_width", "raw_admin_counts_preserved", "missing_category_preserved", "attach_stage"]:
            if key not in bridge:
                failures.append({"kind": f"{name}_race_bridge_missing_key", "key": key})
        if bridge.get("raw_admin_counts_preserved") is not True:
            failures.append({"kind": f"{name}_raw_admin_not_preserved"})
        if bridge.get("missing_category_preserved") is not True:
            failures.append({"kind": f"{name}_missing_not_preserved"})
    source_hashes = manifest.get("source_hashes") or {}
    for key in ["race_bridge_prior", "race_bridge_summary", "sim_processed_events"]:
        if key not in source_hashes:
            failures.append({"kind": "manifest_missing_source_hash", "key": key})
    telemetry = manifest.get("telemetry") or {}
    if telemetry.get("stage_status", {}).get("race_bridge") != "success":
        failures.append({"kind": "race_bridge_stage_not_success", "stage_status": telemetry.get("stage_status", {}).get("race_bridge")})
    q = pl.read_parquet(run / "Q_tensor.parquet")
    for fid in ["SIMRaceBridgePosteriorCount_parda", "SIMRaceBridgeMissingRaceObserver"]:
        row = q.filter(pl.col("field_id") == fid).to_dicts()
        if len(row) != 1:
            failures.append({"kind": "q_row_count", "field_id": fid, "count": len(row)})
            continue
        row = row[0]
        if row.get("dashboard_safe") == "True":
            failures.append({"kind": "dashboard_not_downgraded", "field_id": fid})
        if "sensitivity_width" in q.columns and row.get("sensitivity_width") is None:
            failures.append({"kind": "missing_sensitivity_width", "field_id": fid})
    table = run / "Tables" / "race_bridge_summary.parquet"
    if not table.exists():
        failures.append({"kind": "missing_race_bridge_summary"})
    if failures:
        fail({"status": "failed", "failures": failures})
    print("AUDIT PASSED: Slice 4B registry-backed compile Race Bridge fields validate with raw counts, missing observer, posterior metadata, and telemetry.")


if __name__ == "__main__":
    main()
