from pathlib import Path

import polars as pl

from pegasus.sidra.facts import classify_sidra_value, normalize_fixture_json_to_facts


def test_sidra_value_status_preservation():
    assert classify_sidra_value("10")[1] == "numeric"
    assert classify_sidra_value("-")[1] == "dash_zero_or_nil"
    assert classify_sidra_value("x")[1] == "not_available"
    assert classify_sidra_value("abc")[1] == "non_numeric_symbol"


def test_sidra_fixture_normalizes_to_long_form_facts(tmp_path: Path):
    out = tmp_path / "facts.parquet"
    normalize_fixture_json_to_facts(
        input_path="tests/fixtures/sidra/sidra_flat_fixture.json",
        output_path=out,
        table_id="9606",
        unit_by_variable={"93": "persons"},
    )

    df = pl.read_parquet(out)
    assert df.height == 4
    assert {
        "table_id",
        "variable_id",
        "period",
        "locality_level",
        "locality_id",
        "classification_tuple",
        "category_tuple",
        "value_raw",
        "value_numeric",
        "value_status",
        "unit",
        "request_hash",
        "metadata_hash",
        "fetched_at",
    } == set(df.columns)

    statuses = set(df["value_status"].to_list())
    assert "numeric" in statuses
    assert "dash_zero_or_nil" in statuses
