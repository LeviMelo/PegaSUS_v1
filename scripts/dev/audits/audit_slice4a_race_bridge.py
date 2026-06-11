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


def _json_cell(value):
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


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
    field_ids = set(v["field_id"].to_list())
    required = {
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
    missing = sorted(required - field_ids)
    if missing:
        failures.append({"kind": "missing_race_bridge_fields", "missing": missing})

    for row in v.filter(pl.col("field_id").str.starts_with("SIMRaceBridgePosteriorCount_")).to_dicts():
        axes = _json_cell(row["axes_json"])
        required_metadata = [
            "numerator_axis_source",
            "denominator_axis_target",
            "bridge_operator",
            "emission_matrix_registry_version",
            "bridge_mode",
            "missing_race_share",
            "race_bridge_cv",
            "sensitivity_width",
            "race_axis_warning",
            "bayesian_ecological_bridge_warning",
            "prior_hash",
            "lower_count",
            "upper_count",
        ]
        absent = [key for key in required_metadata if key not in axes]
        if absent:
            failures.append({"kind": "posterior_missing_bridge_metadata", "field_id": row["field_id"], "missing": absent})
        if row["dashboard_safe"] == "True" and float(axes.get("sensitivity_width", 0.0)) > 0.05:
            failures.append({"kind": "posterior_dashboard_safety_not_downgraded", "field_id": row["field_id"], "sensitivity_width": axes.get("sensitivity_width")})

    warnings = pl.read_parquet(run / "Warnings.parquet")
    warning_text = "\n".join(str(x) for x in warnings["message"].to_list()) if "message" in warnings.columns else ""
    if "Raw SIM administrative race/color counts are preserved" not in warning_text:
        failures.append({"kind": "missing_raw_admin_preservation_warning"})
    if "Posterior race counts are bridge-derived observer fields" not in warning_text:
        failures.append({"kind": "missing_bayesian_bridge_warning"})

    q = pl.read_parquet(run / "Q_tensor.parquet")
    q_ids = set(q["field_id"].to_list())
    q_missing = sorted(required - q_ids)
    if q_missing:
        failures.append({"kind": "q_tensor_missing_race_bridge_fields", "missing": q_missing})

    table_path = run / "Tables" / "race_bridge_summary.parquet"
    if not table_path.exists():
        failures.append({"kind": "missing_race_bridge_summary_table", "path": str(table_path)})
    else:
        table = pl.read_parquet(table_path)
        if table.height != 5:
            failures.append({"kind": "invalid_race_bridge_summary_rows", "rows": table.height})
        for column in ["posterior_count", "lower_count", "upper_count", "sensitivity_width", "race_bridge_cv", "missing_race_share"]:
            if column not in table.columns:
                failures.append({"kind": "missing_race_bridge_summary_column", "column": column})

    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
    bridge = run_config.get("race_bridge")
    if not isinstance(bridge, dict):
        failures.append({"kind": "missing_run_config_race_bridge"})
    else:
        for key in ["bridge_id", "mode", "prior_hash", "missing_race_share", "sensitivity_width", "raw_admin_counts_preserved", "missing_category_preserved"]:
            if key not in bridge:
                failures.append({"kind": "run_config_race_bridge_missing_key", "key": key})
        if bridge.get("raw_admin_counts_preserved") is not True:
            failures.append({"kind": "raw_admin_counts_not_marked_preserved"})
        if bridge.get("missing_category_preserved") is not True:
            failures.append({"kind": "missing_category_not_marked_preserved"})

    if failures:
        fail({"status": "failed", "failures": failures})
    print("AUDIT PASSED: Slice 4A race bridge preserves raw admin counts, missing race observer, posterior metadata, sensitivity intervals, and output validation.")


if __name__ == "__main__":
    main()
