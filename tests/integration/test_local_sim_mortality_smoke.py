from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.field_store import read_field_data
from pegasus.workflows.local_smoke import run_local_sim_mortality_smoke


def test_local_sim_mortality_smoke_end_to_end(tmp_path) -> None:
    project_root = Path.cwd()

    raw_sim_path = tmp_path / "sim_raw.csv"
    population_path = tmp_path / "population.csv"

    raw_sim = pl.DataFrame(
        {
            "DTOBITO": ["20220131", "20220201", "20230101", "00000000"],
            "DTNASC": ["19800131", "19700101", "19600101", None],
            "IDADE": ["4042", "4052", "4063", "9001"],
            "SEXO": ["1", "2", "1", "9"],
            "RACACOR": ["4", "1", "2", None],
            "CODMUNRES": ["270430", "270430", "270030", "270430"],
            "CODMUNOCOR": ["270430", "270430", "270030", "270430"],
            "CODESTAB": ["1234567", "1234567", None, None],
            "CAUSABAS": ["I219", "A419", "J189", ""],
            "DIFDATA": ["3", "5", None, None],
            "TPPOS": ["1", "1", None, None],
            "ALTCAUSA": ["2", "2", None, None],
        }
    )

    population = pl.DataFrame(
        {
            "year": [2022, 2023],
            "municipality_cod6": ["270430", "270030"],
            "population": [1000.0, 2000.0],
            "population_state": ["valid", "valid"],
        }
    )

    raw_sim.write_csv(raw_sim_path)
    population.write_csv(population_path)

    result = run_local_sim_mortality_smoke(
        root=project_root,
        intent_path=project_root / "config" / "intents" / "alagoas_smoke.json",
        raw_sim_path=raw_sim_path,
        population_path=population_path,
        normalization_input_kind="raw",
        geography="residence",
        population_source_label="test_population",
    )

    run_dir = Path(result.run_dir)

    assert run_dir.exists()
    assert Path(result.normalized_sim_path).exists()
    assert Path(result.summary_path).exists()

    v_fields = pl.read_parquet(run_dir / "V_fields.parquet")
    e_dag = pl.read_parquet(run_dir / "E_DAG.parquet")
    q_tensor = pl.read_parquet(run_dir / "Q_tensor.parquet")
    warnings = pl.read_parquet(run_dir / "Warnings.parquet")

    assert v_fields.height == 3
    assert q_tensor.height == 3
    assert e_dag.height == 2

    field_ids = set(v_fields["field_id"].to_list())

    assert result.death_count_field_id in field_ids
    assert result.population_field_id in field_ids
    assert result.crude_mortality_field_id in field_ids

    death_data = read_field_data(run_dir, result.death_count_field_id)
    mortality_data = read_field_data(run_dir, result.crude_mortality_field_id)

    assert death_data.to_dicts() == [
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

    assert mortality_data.to_dicts() == [
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

    # One SIM row is intentionally invalid, so we expect warning materialization.
    assert warnings.height >= 1

    with Path(result.summary_path).open("r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "success"
    assert summary["death_count_field_id"] == result.death_count_field_id
    assert summary["population_field_id"] == result.population_field_id
    assert summary["crude_mortality_field_id"] == result.crude_mortality_field_id