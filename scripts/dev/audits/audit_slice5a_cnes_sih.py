from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl
from pegasus.output.validate import validate_output_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    run = Path(parser.parse_args().run)
    result = validate_output_bundle(run_dir=str(run))
    if not result.ok:
        raise SystemExit("Output validation failed: " + "; ".join(result.errors))
    v = pl.read_parquet(run / "V_fields.parquet")
    failed = pl.read_parquet(run / "FailedBranches.parquet")
    warnings = pl.read_parquet(run / "Warnings.parquet")
    names = set(v["name"].to_list())
    required = {"CNESFacilitiesAll", "CNESCapacity_QTLEITP1", "CNESCapacity_QTLEITP2", "CNESCapacity_QTLEITP3", "CNESInvalidBooleanFlagShare", "CNESZeroFacilityCNPJShare", "SIHHospitalAdmissionsAll", "SIHInpatientDeaths", "SIHInpatientFatality", "SIHMeanLengthOfStay", "SIHCost_VAL_SH", "SIHCost_VAL_SP", "SIHCost_VAL_UTI", "SIHCost_VAL_TOT"}
    if required - names:
        raise SystemExit(f"Missing required Slice 5A fields: {sorted(required - names)}")
    failed_ids = set(failed["failed_branch_id"].to_list())
    for branch in ["failed_cnes_generic_beds_without_capacity_index", "failed_facility_flow_unsanitized_cnpj", "failed_sih_generic_cost_pooling", "failed_sih_diagnostic_topology_collapse"]:
        if branch not in failed_ids:
            raise SystemExit(f"Missing failed branch: {branch}")
    warning_codes = set(warnings["code"].to_list())
    for code in ["generic_beds_forbidden_without_capacity_index", "all_zero_cnpj_nullified", "generic_sih_cost_forbidden", "sih_principal_secondary_diagnosis_topology_preserved"]:
        if code not in warning_codes:
            raise SystemExit(f"Missing warning code: {code}")
    print("AUDIT PASSED: Slice 5A CNES-ST/SIH-RD source-realistic normalization, indexed CNES capacity, SIH component costs, and hard failed branches validated.")


if __name__ == "__main__":
    main()
