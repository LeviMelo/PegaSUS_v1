from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.sinasc import run_build_sinasc_fixture, run_datasus_normalize_sinasc


def test_slice3a_sinasc_normalize_to_maternal_child_bundle(tmp_path: Path):
    events = tmp_path / "processed" / "sinasc_events.parquet"
    normalize = run_datasus_normalize_sinasc(
        input_path="tests/fixtures/datasus/sinasc_fixture.csv",
        output_path=events,
        source_manifest_hash="fixture_manifest",
    )
    assert normalize["row_count"] == 5

    result = run_build_sinasc_fixture(
        sinasc_events_path=events,
        run_dir=tmp_path / "run",
    )
    assert result["validation"].ok, result["validation"].errors
    assert validate_output_bundle(run_dir=str(result["run_dir"])).ok

    table = pl.read_parquet(result["run_dir"] / "Tables" / "sinasc_maternal_child_summary.parquet")
    assert table["births_total"].item() == 5
    assert table["low_birth_weight_births"].item() == 2
    assert table["congenital_anomaly_births"].item() == 4
    assert table["insufficient_prenatal_births"].item() == 2


def test_slice3a_sinasc_municipality_filter_changes_support(tmp_path: Path):
    events = tmp_path / "processed" / "sinasc_events.parquet"
    run_datasus_normalize_sinasc(
        input_path="tests/fixtures/datasus/sinasc_fixture.csv",
        output_path=events,
        source_manifest_hash="fixture_manifest",
    )
    result = run_build_sinasc_fixture(
        sinasc_events_path=events,
        run_dir=tmp_path / "run_maceio",
        municipality_cod6="270430",
    )
    assert result["validation"].ok, result["validation"].errors
    table = pl.read_parquet(result["run_dir"] / "Tables" / "sinasc_maternal_child_summary.parquet")
    assert table["births_total"].item() == 3
    assert table["low_birth_weight_births"].item() == 1
    assert table["congenital_anomaly_births"].item() == 3
