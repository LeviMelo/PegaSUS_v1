from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.output.validate import validate_output_bundle


REQUIRED_FIELDS = {
    "cnes_facilities_all",
    "cnes_capacity_qtinst34",
    "cnes_zero_facility_cnpj_share",
    "cnes_invalid_boolean_flag_share",
    "sih_admissions_all",
    "sih_inpatient_deaths",
    "sih_inpatient_fatality",
    "sih_mean_los",
    "sih_cost_val_sh",
    "sih_cost_val_sp",
    "sih_cost_val_uti",
    "sih_cost_val_tot",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)

    validation = validate_output_bundle(run_dir=str(run))
    if not validation.ok:
        raise SystemExit("Output validation failed: " + "; ".join(validation.errors))

    fields = pq.read_table(run / "V_fields.parquet").to_pylist()
    ids = {str(row["field_id"]) for row in fields}
    missing = sorted(REQUIRED_FIELDS - ids)
    if missing:
        raise SystemExit(f"Missing Slice 5B CNES/SIH fields: {missing}")

    by_id = {str(row["field_id"]): row for row in fields}
    cnes_axes = json.loads(by_id["cnes_capacity_qtinst34"]["axes_json"])
    if cnes_axes.get("capacity_vector_index") != "QTINST34":
        raise SystemExit("CNES capacity field lost capacity_vector_index metadata")
    if cnes_axes.get("generic_beds_forbidden") is not True:
        raise SystemExit("CNES capacity field does not block generic bed semantics")

    sih_axes = json.loads(by_id["sih_cost_val_tot"]["axes_json"])
    if sih_axes.get("cost_component") != "VAL_TOT":
        raise SystemExit("SIH cost field lost VAL_TOT component metadata")
    if sih_axes.get("generic_sih_cost_forbidden") is not True:
        raise SystemExit("SIH cost field does not block generic cost semantics")

    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    if run_config.get("cnes_sih", {}).get("cnes", {}).get("generic_beds_blocked") is not True:
        raise SystemExit("RunConfig.cnes_sih does not record generic CNES bed block")
    if manifest.get("cnes_sih", {}).get("sih", {}).get("diagnostic_topology_preserved") is not True:
        raise SystemExit("ReproducibilityManifest.cnes_sih does not record SIH diagnostic topology preservation")

    for name in ["slice5a_cnes_capacity_summary.parquet", "slice5a_sih_cost_summary.parquet"]:
        if not (run / "Tables" / name).exists():
            raise SystemExit(f"Missing summary table: {name}")

    print("AUDIT PASSED: Slice 5B compile integrates CNES-ST capacity vectors and SIH-RD component-specific hospitalization/cost fields with validator-enforced metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
