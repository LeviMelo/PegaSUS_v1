from pathlib import Path

import polars as pl

from pegasus.datasus.sinasc_normalize import normalize_sinasc_events
from pegasus.output.sinasc_efg_bundle import write_sinasc_fixture_efg_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.she.maternal_child import summarize_maternal_child_events


def _events(tmp_path: Path) -> Path:
    out = tmp_path / "sinasc_events.parquet"
    normalize_sinasc_events(
        input_path="tests/fixtures/datasus/sinasc_fixture.csv",
        output_path=out,
        source_manifest_hash="fixture_manifest",
    )
    return out


def test_maternal_child_summary_counts_fixture_events(tmp_path: Path):
    summary = summarize_maternal_child_events(_events(tmp_path))
    assert summary.births_total == 5
    assert summary.low_birth_weight_births == 2
    assert summary.prematurity_births == 2
    assert summary.cesarean_births == 2
    assert summary.congenital_anomaly_births == 2
    assert summary.low_apgar5_births == 1
    assert summary.adolescent_mother_births == 1
    assert summary.advanced_maternal_age_births == 1
    assert summary.insufficient_prenatal_births == 2
    assert summary.years == [2022]
    assert summary.municipalities_cod6 == ["270030", "270430"]


def test_sinasc_efg_bundle_validates_and_blocks_missing_denominators_and_mortality(tmp_path: Path):
    events = _events(tmp_path)
    run_dir = write_sinasc_fixture_efg_bundle(sinasc_events_path=events, run_dir=tmp_path / "run")
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    names = set(v["field_id"].to_list())
    assert "sinasc_births_all" in names
    assert "sinasc_low_birth_weight_prevalence" in names
    assert "sinasc_congenital_anomaly_prevalence" in names
    assert "sinasc_low_apgar5_prevalence" in names
    assert "sinasc_insufficient_prenatal_share" in names

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    assert set(v["field_id"].to_list()).issubset(set(q["field_id"].to_list()))

    failed = pl.read_parquet(run_dir / "FailedBranches.parquet")
    reasons = set(failed["reason"].to_list())
    assert "blocked_missing_population_denominator_anchor" in reasons
    assert "blocked_missing_sim_death_numerator_linkage" in reasons
    assert "blocked_missing_sim_neonatal_death_numerator_linkage" in reasons
    assert "blocked_missing_sim_postneonatal_death_numerator_linkage" in reasons
