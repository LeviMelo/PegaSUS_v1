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
    field_ids = set(v["field_id"].to_list())
    required = {
        "sinasc_births_all",
        "sinasc_low_birth_weight_births",
        "sinasc_prematurity_births",
        "sinasc_cesarean_births",
        "sinasc_congenital_anomaly_births",
        "sinasc_low_apgar5_births",
        "sinasc_adolescent_mother_births",
        "sinasc_advanced_maternal_age_births",
        "sinasc_insufficient_prenatal_births",
        "sinasc_low_birth_weight_prevalence",
        "sinasc_prematurity_prevalence",
        "sinasc_cesarean_prevalence",
        "sinasc_congenital_anomaly_prevalence",
        "sinasc_low_apgar5_prevalence",
        "sinasc_adolescent_mother_share",
        "sinasc_advanced_maternal_age_share",
        "sinasc_insufficient_prenatal_share",
    }
    missing = sorted(required - field_ids)
    if missing:
        failures.append({"kind": "missing_fields", "missing": missing})

    failed = pl.read_parquet(run / "FailedBranches.parquet")
    reasons = set(failed["reason"].to_list()) if "reason" in failed.columns else set()
    for reason in [
        "blocked_missing_population_denominator_anchor",
        "blocked_missing_sim_death_numerator_linkage",
        "blocked_missing_sim_neonatal_death_numerator_linkage",
        "blocked_missing_sim_postneonatal_death_numerator_linkage",
    ]:
        if reason not in reasons:
            failures.append({"kind": "missing_failed_branch", "reason": reason})

    warnings = pl.read_parquet(run / "Warnings.parquet")
    messages = "\n".join(str(x) for x in warnings["message"].to_list()) if "message" in warnings.columns else ""
    if "Maternal and newborn administrative race axes remain separate" not in messages:
        failures.append({"kind": "missing_race_axis_warning", "messages": messages})
    if "Crude birth rate requires a legal population denominator" not in messages:
        failures.append({"kind": "missing_birth_rate_denominator_warning", "messages": messages})

    table_path = run / "Tables" / "sinasc_maternal_child_summary.parquet"
    if not table_path.exists():
        failures.append({"kind": "missing_summary_table", "path": str(table_path)})
    else:
        summary = pl.read_parquet(table_path)
        if summary["births_total"].item() <= 0:
            failures.append({"kind": "invalid_birth_count", "births_total": summary["births_total"].item()})
        for col in ["low_birth_weight_prevalence", "prematurity_prevalence", "congenital_anomaly_prevalence"]:
            val = summary[col].item()
            if val is None or val < 0 or val > 1:
                failures.append({"kind": "invalid_prevalence", "column": col, "value": val})

    if failures:
        fail({"status": "failed", "failures": failures})

    print("AUDIT PASSED: Slice 3A SINASC maternal-child fields validate, preserve race-axis warning, and block mortality/birth-rate denominators until legal linkage exists.")


if __name__ == "__main__":
    main()
