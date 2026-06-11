from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle


REQUIRED_FIELDS = {
    "SINASCLiveBirthsAll",
    "SINASCCrudeBirthRateSIDRAOfficial",
    "SINASCLowBirthWeightPrevalence",
    "SINASCPrematurityPrevalence",
    "SINASCCongenitalAnomalyPrevalence",
    "SIMInfantMortalitySINASCBirths",
    "SIMNeonatalMortalitySINASCBirths",
    "SIMPostNeonatalMortalitySINASCBirths",
}


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
    field_ids = set(str(x) for x in v["field_id"].to_list())
    missing = sorted(REQUIRED_FIELDS - field_ids)
    if missing:
        failures.append({"kind": "missing_required_fields", "missing": missing})

    q = pl.read_parquet(run / "Q_tensor.parquet")
    q_ids = set(str(x) for x in q["field_id"].to_list())
    missing_q = sorted(REQUIRED_FIELDS - q_ids)
    if missing_q:
        failures.append({"kind": "missing_q_rows", "missing": missing_q})

    vd = pl.read_parquet(run / "VariableDictionary.parquet")
    vd_ids = set(str(x) for x in vd["field_id"].to_list())
    missing_vd = sorted(REQUIRED_FIELDS - vd_ids)
    if missing_vd:
        failures.append({"kind": "missing_variable_dictionary_rows", "missing": missing_vd})

    table_path = run / "Tables" / "maternal_child_linkage_summary.parquet"
    if not table_path.exists():
        failures.append({"kind": "missing_summary_table", "path": str(table_path)})
    else:
        table = pl.read_parquet(table_path)
        births_total = table["births_total"].item()
        denom = table["denominator_population"].item()
        if births_total <= 0:
            failures.append({"kind": "invalid_births_total", "births_total": births_total})
        if denom <= 0:
            failures.append({"kind": "invalid_population_denominator", "denominator_population": denom})

    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
    source_hashes = run_config.get("source_hashes", {})
    for key in ["sinasc_raw_fixture", "sinasc_processed_events", "sim_processed_events", "sidra_facts"]:
        if key not in source_hashes:
            failures.append({"kind": "missing_source_hash", "key": key})
    linkage = run_config.get("maternal_child_linkage", {})
    if linkage.get("enabled") is not True:
        failures.append({"kind": "missing_maternal_child_linkage_run_config", "value": linkage})

    if failures:
        fail({"status": "failed", "failures": failures})
    print("AUDIT PASSED: Slice 3B compile run contains SINASC crude birth, maternal-child prevalence, and SIM/SINASC mortality linkage fields with validated output bundle metadata.")


if __name__ == "__main__":
    main()
