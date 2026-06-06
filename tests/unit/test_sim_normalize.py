from pathlib import Path

import polars as pl

from pegasus.datasus.normalize import SIM_DO_NORMALIZED_COLUMNS, normalize_sim_do_events


def test_sim_do_fixture_normalization_schema(tmp_path: Path):
    output = tmp_path / "sim_events.parquet"

    result = normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=output,
        source_manifest_hash="fixture_manifest_hash",
    )

    assert result["row_count"] == 3
    assert output.exists()

    df = pl.read_parquet(output)
    assert df.columns == SIM_DO_NORMALIZED_COLUMNS
    assert df.height == 3


def test_sim_do_age_and_icd_topology_normalization(tmp_path: Path):
    output = tmp_path / "sim_events.parquet"

    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=output,
        source_manifest_hash="fixture_manifest_hash",
    )

    df = pl.read_parquet(output).sort("death_date")

    first = df.row(0, named=True)

    # Date-derived age is preferred when DTNASC and DTOBITO are both valid.
    # The raw SIM structural age code is still preserved as provenance.
    assert first["age_source"] == "date_difference"
    assert abs(first["age_years"] - 74.0) < 0.01
    assert first["raw_age_code"] == "474"
    assert first["age_unit"] == "years"

    assert first["underlying_icd_norm"] == "A419"
    assert first["underlying_icd_parse_state"] == "valid"
    assert '"A": "*J189"' in first["cause_chain_raw"]
    assert '"A": "J189"' in first["cause_chain_norm"]

    second = df.row(1, named=True)
    assert second["age_source"] == "IDADE"
    assert second["underlying_icd_norm"] == "R99"
    assert second["underlying_icd_parse_state"] == "ill-defined"
    assert second["race_missingness_state"] == "unknown"


def test_sim_do_missing_and_sentinel_values_are_not_silent_negatives(tmp_path: Path):
    output = tmp_path / "sim_events.parquet"

    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=output,
        source_manifest_hash="fixture_manifest_hash",
    )

    df = pl.read_parquet(output).sort("death_date")

    second = df.row(1, named=True)
    assert second["maternal_living_children_count"] is None
    assert second["birth_weight_death_context_grams"] is None
    assert second["facility_code"] is None
    assert second["facility_code_state"] == "missing"

    third = df.row(2, named=True)
    assert third["death_hour"] is None
    assert third["associated_conditions_norm"] == "Q249"
    assert third["associated_conditions_parse_states"] == "valid"
