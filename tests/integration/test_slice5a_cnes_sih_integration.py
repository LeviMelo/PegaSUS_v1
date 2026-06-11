from pathlib import Path
import polars as pl
from pegasus.workflows.cnes_sih import run_build_cnes_sih_fixture, run_datasus_normalize_cnes, run_datasus_normalize_sih


def test_slice5a_cnes_sih_bundle_validates_and_preserves_contracts(tmp_path: Path):
    cnes = tmp_path / "cnes.parquet"
    sih = tmp_path / "sih.parquet"
    run = tmp_path / "run"
    run_datasus_normalize_cnes(input_path="tests/fixtures/datasus/cnes_st_fixture.csv", output_path=cnes, source_manifest_hash="fixture")
    run_datasus_normalize_sih(input_path="tests/fixtures/datasus/sih_rd_fixture.csv", output_path=sih, source_manifest_hash="fixture")
    result = run_build_cnes_sih_fixture(cnes_events_path=cnes, sih_events_path=sih, run_dir=run, municipality_cod6="270430")
    assert result["validation"].ok, result["validation"].errors
    names = set(pl.read_parquet(run / "V_fields.parquet")["name"].to_list())
    for name in ["CNESCapacity_QTLEITP1", "CNESCapacity_QTLEITP2", "CNESCapacity_QTLEITP3", "CNESInvalidBooleanFlagShare", "CNESZeroFacilityCNPJShare", "SIHHospitalAdmissionsAll", "SIHInpatientFatality", "SIHCost_VAL_SH", "SIHCost_VAL_SP", "SIHCost_VAL_UTI", "SIHCost_VAL_TOT"]:
        assert name in names
    failed_ids = set(pl.read_parquet(run / "FailedBranches.parquet")["failed_branch_id"].to_list())
    for branch in ["failed_cnes_generic_beds_without_capacity_index", "failed_facility_flow_unsanitized_cnpj", "failed_sih_generic_cost_pooling", "failed_sih_diagnostic_topology_collapse"]:
        assert branch in failed_ids
