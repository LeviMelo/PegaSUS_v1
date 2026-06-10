from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle

REQUIRED_SUCCESS_STAGES = {
    "datasus_manifest",
    "datasus_acquire",
    "datasus_decode",
    "sidra_normalize",
    "efg_build",
    "geo_support",
    "she_build",
    "q_tensor",
    "output_serialization",
    "output_validation",
}

REQUIRED_BLOCKED_STAGES = {"population_solver", "stdfm", "pirs_model", "pirs_hsic"}

REQUIRED_SOURCE_HASHES = {
    "intent",
    "compile_manifest",
    "sim_raw_fixture",
    "sim_processed_events",
    "sidra_facts",
}

REQUIRED_FIELDS = {
    "SIMDeathsAll",
    "SIDRAPopulationTotalAnchor",
    "SIMCrudeMortalitySIDRAOfficial",
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

    try:
        intent = json.loads((run / "UserIntent.json").read_text(encoding="utf-8"))
        run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
        manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        fail({"status": "failed", "reason": "json_read_failure", "error": str(exc)})

    if intent.get("execution_scale") != "smoke":
        failures.append({"kind": "intent", "expected": "execution_scale=smoke", "actual": intent.get("execution_scale")})
    if run_config.get("compile_mode") != "smoke":
        failures.append({"kind": "run_config", "expected": "compile_mode=smoke", "actual": run_config.get("compile_mode")})

    telemetry = manifest.get("telemetry", {})
    statuses = telemetry.get("stage_status", {})
    durations = telemetry.get("stage_wall_seconds", {})

    for stage in REQUIRED_SUCCESS_STAGES:
        if statuses.get(stage) != "success":
            failures.append({"kind": "telemetry_success_stage", "stage": stage, "actual": statuses.get(stage)})
        if durations.get(stage, -1) < 0:
            failures.append({"kind": "telemetry_duration", "stage": stage, "actual": durations.get(stage)})

    for stage in REQUIRED_BLOCKED_STAGES:
        if statuses.get(stage) != "blocked":
            failures.append({"kind": "telemetry_blocked_stage", "stage": stage, "actual": statuses.get(stage)})

    source_hashes = manifest.get("source_hashes", {})
    missing_source_hashes = sorted(REQUIRED_SOURCE_HASHES - set(source_hashes))
    if missing_source_hashes:
        failures.append({"kind": "missing_source_hashes", "missing": missing_source_hashes})

    if not manifest.get("registry_hashes"):
        failures.append({"kind": "missing_registry_hashes"})

    try:
        v = pl.read_parquet(run / "V_fields.parquet")
        names = set(v["name"].to_list())
        missing_fields = sorted(REQUIRED_FIELDS - names)
        if missing_fields:
            failures.append({"kind": "missing_fields", "missing": missing_fields})

        rate_rows = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").to_dicts()
        if len(rate_rows) != 1:
            failures.append({"kind": "rate_field_count", "expected": 1, "actual": len(rate_rows)})
        else:
            support = json.loads(rate_rows[0]["support_json"])
            alignment = support.get("support_alignment", {})
            if alignment.get("aligned") is not True:
                failures.append({"kind": "support_alignment", "alignment": alignment})
            if alignment.get("numerator_municipalities_ibge_cod7") != alignment.get("denominator_municipalities_ibge_cod7"):
                failures.append({"kind": "support_alignment_municipalities", "alignment": alignment})
    except Exception as exc:
        failures.append({"kind": "field_read_failure", "error": str(exc)})

    if failures:
        fail({"status": "failed", "failures": failures})

    print("AUDIT PASSED: Slice 2D compile smoke emitted validated telemetry, provenance, SIDRA denominator, and support alignment.")


if __name__ == "__main__":
    main()
