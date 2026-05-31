from __future__ import annotations

import polars as pl

from pegasus.datasus.normalize_sim import normalize_sim_do
from pegasus.output.bundle import OutputBundle
from pegasus.output.field_store import read_field_data
from pegasus.output.write_tables import write_compiled_field
from pegasus.problem1.population_fields import compile_imported_population_field
from pegasus.problem1.rate_fields import compile_crude_mortality_rate_field
from pegasus.problem1.sim_fields import compile_sim_death_count_field
from pegasus.registries.index import RegistryIndex
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set


def _registry(tmp_path):
    root = tmp_path.parents[0]
    # Test file lives under tests/unit; project root is cwd under pytest.
    from pathlib import Path

    project_root = Path.cwd()
    registry_set = load_registry_set(project_root / "config" / "registries")
    validate_registry_set(registry_set)
    return registry_set, RegistryIndex.from_registry_set(registry_set)


def test_population_denominator_and_crude_mortality_rate_compile(tmp_path) -> None:
    registry_set, registry_index = _registry(tmp_path)

    raw_sim = pl.DataFrame(
        {
            "DTOBITO": ["20220131", "20220201", "20230101"],
            "IDADE": ["4042", "4050", "4060"],
            "SEXO": ["1", "2", "1"],
            "RACACOR": ["4", "1", "2"],
            "CODMUNRES": ["270430", "270430", "270030"],
            "CAUSABAS": ["I219", "A419", "J189"],
        }
    )

    normalized = normalize_sim_do(raw_sim, source_manifest_hash="sim_test")

    death_compiled = compile_sim_death_count_field(
        normalized,
        registry_hashes=registry_set.hashes(),
        geography="residence",
    )

    population_df = pl.DataFrame(
        {
            "year": [2022, 2023],
            "municipality_cod6": ["270430", "270030"],
            "population": [1_000.0, 2_000.0],
            "population_state": ["valid", "valid"],
        }
    )

    population_compiled = compile_imported_population_field(
        population_df,
        registry_hashes=registry_set.hashes(),
        source_label="test_population",
    )

    mortality_compiled = compile_crude_mortality_rate_field(
        death_count_field=death_compiled.field,
        death_count_data=death_compiled.data,
        population_field=population_compiled.field,
        population_data=population_compiled.data,
        registry_index=registry_index,
        registry_hashes=registry_set.hashes(),
        scale=100_000.0,
    )

    rows = mortality_compiled.data.to_dicts()

    assert rows == [
        {
            "year": 2022,
            "municipality_cod6": "270430",
            "death_count": 2.0,
            "population": 1000.0,
            "mortality_rate": 0.002,
            "mortality_rate_scaled": 200.0,
            "scale": 100000.0,
            "unit": "deaths_per_100k_person_years",
        },
        {
            "year": 2023,
            "municipality_cod6": "270030",
            "death_count": 1.0,
            "population": 2000.0,
            "mortality_rate": 0.0005,
            "mortality_rate_scaled": 50.0,
            "scale": 100000.0,
            "unit": "deaths_per_100k_person_years",
        },
    ]

    assert mortality_compiled.field.kind == "intensive_density"
    assert mortality_compiled.field.carrier == "Deaths/Population"
    assert mortality_compiled.field.unit == "deaths_per_100k_person_years"
    assert mortality_compiled.field.lineage.parent_ids == [
        death_compiled.field.id,
        population_compiled.field.id,
    ]


def test_bundle_receives_death_population_and_mortality_fields(tmp_path) -> None:
    registry_set, registry_index = _registry(tmp_path)

    run_dir = tmp_path / "run"
    bundle = OutputBundle(run_dir)
    bundle.initialize()
    bundle.write_user_intent({"name": "test"})
    bundle.write_run_config({"project": {"name": "PegaSUS"}})
    bundle.write_p_vector({})
    bundle.write_reproducibility_manifest("run", registry_hashes=registry_set.hashes())
    bundle.validate_minimal()

    raw_sim = pl.DataFrame(
        {
            "DTOBITO": ["20220131", "20220201"],
            "IDADE": ["4042", "4050"],
            "SEXO": ["1", "2"],
            "RACACOR": ["4", "1"],
            "CODMUNRES": ["270430", "270430"],
            "CAUSABAS": ["I219", "A419"],
        }
    )

    normalized = normalize_sim_do(raw_sim, source_manifest_hash="sim_test")

    death = compile_sim_death_count_field(
        normalized,
        registry_hashes=registry_set.hashes(),
    )

    population = compile_imported_population_field(
        pl.DataFrame(
            {
                "year": [2022],
                "municipality_cod6": ["270430"],
                "population": [1000.0],
            }
        ),
        registry_hashes=registry_set.hashes(),
    )

    death_field = write_compiled_field(run_dir, death)
    population_field = write_compiled_field(run_dir, population)

    mortality = compile_crude_mortality_rate_field(
        death_count_field=death_field,
        death_count_data=death.data,
        population_field=population_field,
        population_data=population.data,
        registry_index=registry_index,
        registry_hashes=registry_set.hashes(),
    )

    mortality_field = write_compiled_field(run_dir, mortality)

    v_fields = pl.read_parquet(run_dir / "V_fields.parquet")
    e_dag = pl.read_parquet(run_dir / "E_DAG.parquet")
    q_tensor = pl.read_parquet(run_dir / "Q_tensor.parquet")

    assert v_fields.height == 3
    assert q_tensor.height == 3
    assert e_dag.height == 2

    mortality_data = read_field_data(run_dir, mortality_field.id)
    assert mortality_data.height == 1
    assert mortality_data.row(0, named=True)["mortality_rate_scaled"] == 200.0