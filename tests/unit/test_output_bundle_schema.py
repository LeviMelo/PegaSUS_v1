from __future__ import annotations

import json

import polars as pl

from pegasus.output.bundle import OutputBundle
from pegasus.output.schemas import PARQUET_FILENAMES, PARQUET_SCHEMAS


def test_output_bundle_initializes_all_required_schema_tables(tmp_path) -> None:
    run_dir = tmp_path / "run_test"
    bundle = OutputBundle(run_dir)

    bundle.initialize()
    bundle.write_user_intent({"name": "test"})
    bundle.write_run_config({"config": "test"})
    bundle.write_p_vector({})
    bundle.write_reproducibility_manifest("run_test")
    bundle.validate_minimal()

    for key, schema in PARQUET_SCHEMAS.items():
        path = run_dir / PARQUET_FILENAMES[key]
        assert path.exists(), f"{key} did not exist at {path}"

        df = pl.read_parquet(path)
        assert list(df.columns) == list(schema.keys())

        for column, dtype in schema.items():
            assert df.schema[column] == dtype


def test_output_bundle_writes_json_contract_files(tmp_path) -> None:
    run_dir = tmp_path / "run_test"
    bundle = OutputBundle(run_dir)

    bundle.initialize()
    bundle.write_user_intent({"name": "test_intent"})
    bundle.write_run_config({"project": {"name": "PegaSUS"}})
    bundle.write_p_vector({"field_x": {"provenance": ["official"]}})
    bundle.write_reproducibility_manifest("run_test")
    bundle.validate_minimal()

    with (run_dir / "UserIntent.json").open("r", encoding="utf-8") as f:
        user_intent = json.load(f)

    with (run_dir / "RunConfig.json").open("r", encoding="utf-8") as f:
        run_config = json.load(f)

    with (run_dir / "P_vector.json").open("r", encoding="utf-8") as f:
        p_vector = json.load(f)

    assert user_intent["name"] == "test_intent"
    assert run_config["project"]["name"] == "PegaSUS"
    assert p_vector["field_x"]["provenance"] == ["official"]