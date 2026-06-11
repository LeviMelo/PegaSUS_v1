from pathlib import Path

import polars as pl

from pegasus.datasus.sinasc_normalize import normalize_sinasc_events


def test_sinasc_fixture_normalization_preserves_decoder_states(tmp_path: Path):
    out = tmp_path / "sinasc_events.parquet"
    result = normalize_sinasc_events(
        input_path="tests/fixtures/datasus/sinasc_fixture.csv",
        output_path=out,
        source_manifest_hash="fixture_manifest",
    )
    assert result["row_count"] == 5
    assert result["valid_rows"] == 5
    assert result["low_birth_weight_rows"] == 2
    assert result["prematurity_rows"] == 2
    assert result["cesarean_rows"] == 2
    assert result["low_apgar5_rows"] == 1
    assert result["insufficient_prenatal_rows"] == 2
    assert result["anomaly_rows"] == 2

    df = pl.read_parquet(out)
    assert "raw_json" in df.columns
    assert df.filter(pl.col("event_id") == "SINASC-DN0002")["birth_weight_g"].item() == 2400
    assert df.filter(pl.col("event_id") == "SINASC-DN0002")["low_birth_weight_flag"].item() is True
    assert df.filter(pl.col("event_id") == "SINASC-DN0004")["birth_weight_state"].item() == "missing"
    assert df.filter(pl.col("event_id") == "SINASC-DN0004")["gestational_age_state"].item() == "sentinel"
    assert df.filter(pl.col("event_id") == "SINASC-DN0001")["prenatal_consult_count"].item() == 7
    assert df.filter(pl.col("event_id") == "SINASC-DN0001")["prenatal_consult_raw_digits"].item() == "07"
    assert df.filter(pl.col("event_id") == "SINASC-DN0003")["low_apgar5_flag"].item() is True
    assert df.filter(pl.col("event_id") == "SINASC-DN0005")["advanced_maternal_age_flag"].item() is True
