from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.datasus.io import read_table
from pegasus.output.bundle import OutputBundle
from pegasus.output.write_tables import write_compiled_field
from pegasus.problem1.population_fields import compile_imported_population_field
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set


def import_population_to_bundle(
    *,
    population_path: str | Path,
    run_dir: str | Path,
    registry_dir: str | Path,
    source_label: str = "imported_population_fixture",
) -> str:
    run_dir = Path(run_dir)
    registry_dir = Path(registry_dir)

    bundle = OutputBundle(run_dir)
    bundle.validate_minimal()

    registry_set = load_registry_set(registry_dir)
    validate_registry_set(registry_set)

    population = read_table(population_path)

    compiled = compile_imported_population_field(
        population,
        registry_hashes=registry_set.hashes(),
        source_label=source_label,
    )

    field = write_compiled_field(run_dir, compiled)
    bundle.validate_minimal()

    return field.id