from __future__ import annotations

import polars as pl

from pegasus.datasus.normalize_sim import normalize_sim_do
from pegasus.output.bundle import OutputBundle
from pegasus.output.write_tables import write_compiled_field
from pegasus.problem1.sim_fields import compile_sim_death_count_field


def test_compile_sim_death_count_field_groups_by_year_and_municipality() -> None:
    raw = pl.DataFrame(
        {
            "DTOBITO": ["20220131", "20220201", "20230101", "00000000"],
            "IDADE": ["4042", "4050", "4060", "9001"],
            "SEXO": ["1", "2", "1", "9"],
            "RACACOR": ["4", "1", "2", None],
            "CODMUNRES": ["270430", "270430", "270030", "270430"],
            "CODMUNOCOR": ["270430", "270430", "270030", "270430"],
            "CAUSABAS": ["I219", "A419", "J189", ""],
        }
    )

    normalized = normalize_sim_do(raw, source_manifest_hash="test")
    compiled = compile_sim_death_count_field(
        normalized,
        registry_hashes={"registry_manifest": "abc"},
        geography="residence",
    )

    assert compiled.field.carrier == "Deaths"
    assert compiled.field.unit == "counts"
    assert compiled.field.aggregation == "additive"
    assert compiled.q_state.n_events == 3.0
    assert compiled.q_state.missingness == 0.25

    rows = compiled.data.to_dicts()

    assert rows == [
        {
            "source_system": "SIM-DO",
            "geography_basis": "municipality_residence",
            "year": 2022,
            "municipality_cod6": "270430",
            "death_count": 2,
        },
        {
            "source_system": "SIM-DO",
            "geography_basis": "municipality_residence",
            "year": 2023,
            "municipality_cod6": "270030",
            "death_count": 1,
        },
    ]


def test_write_compiled_field_updates_bundle_tables(tmp_path) -> None:
    run_dir = tmp_path / "run"
    bundle = OutputBundle(run_dir)
    bundle.initialize()
    bundle.write_user_intent({"name": "test"})
    bundle.write_run_config({"project": {"name": "PegaSUS"}})
    bundle.write_p_vector({})
    bundle.write_reproducibility_manifest("run")
    bundle.validate_minimal()

    raw = pl.DataFrame(
        {
            "DTOBITO": ["20220131"],
            "IDADE": ["4042"],
            "SEXO": ["1"],
            "RACACOR": ["4"],
            "CODMUNRES": ["270430"],
            "CAUSABAS": ["I219"],
        }
    )

    normalized = normalize_sim_do(raw, source_manifest_hash="test")
    compiled = compile_sim_death_count_field(
        normalized,
        registry_hashes={"registry_manifest": "abc"},
    )

    field = write_compiled_field(run_dir, compiled)

    assert (run_dir / field.path).exists()

    v_fields = pl.read_parquet(run_dir / "V_fields.parquet")
    q_tensor = pl.read_parquet(run_dir / "Q_tensor.parquet")

    assert v_fields.height == 1
    assert q_tensor.height == 1
    assert v_fields.row(0, named=True)["field_id"] == field.id
    assert q_tensor.row(0, named=True)["field_id"] == field.id