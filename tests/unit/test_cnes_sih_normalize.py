from pathlib import Path
import polars as pl
from pegasus.datasus.cnes_normalize import normalize_cnes_st_events
from pegasus.datasus.sih_normalize import normalize_sih_rd_events


def test_cnes_normalization_preserves_capacity_flags_and_cnpj(tmp_path: Path):
    out = tmp_path / "cnes.parquet"
    result = normalize_cnes_st_events(input_path="tests/fixtures/datasus/cnes_st_fixture.csv", output_path=out, source_manifest_hash="fixture")
    df = pl.read_parquet(out)
    assert result["row_count"] == 3
    assert result["zero_facility_cnpj_rows"] == 1
    assert result["invalid_flag_rows"] == 2
    assert "QTLEITP3" in result["capacity_components"]
    assert "capacity_vector_json" in df.columns
    assert "flag_state_json" in df.columns


def test_sih_normalization_preserves_diagnostic_topology_and_cost_components(tmp_path: Path):
    out = tmp_path / "sih.parquet"
    result = normalize_sih_rd_events(input_path="tests/fixtures/datasus/sih_rd_fixture.csv", output_path=out, source_manifest_hash="fixture")
    df = pl.read_parquet(out)
    assert result["row_count"] == 3
    assert result["deaths"] == 1
    assert set(result["cost_components"]) == {"VAL_SH", "VAL_SP", "VAL_UTI", "VAL_TOT"}
    assert "principal_icd_norm" in df.columns
    assert "secondary_icd_norm_json" in df.columns
    assert "hospital_service_cost_real" in df.columns
    assert "professional_service_cost_real" in df.columns
    assert "icu_cost_real" in df.columns
    assert "total_admission_cost_real" in df.columns
