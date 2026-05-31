from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from pegasus.output.bundle import OutputBundle
from pegasus.output.write_tables import write_compiled_field
from pegasus.problem1.sim_fields import compile_sim_death_count_field
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set


def compile_sim_death_counts_to_bundle(
    *,
    normalized_path: str | Path,
    run_dir: str | Path,
    registry_dir: str | Path,
    geography: Literal["residence", "occurrence"] = "residence",
) -> str:
    run_dir = Path(run_dir)
    normalized_path = Path(normalized_path)
    registry_dir = Path(registry_dir)

    bundle = OutputBundle(run_dir)
    bundle.validate_minimal()

    registry_set = load_registry_set(registry_dir)
    validate_registry_set(registry_set)

    normalized = pl.read_parquet(normalized_path)

    compiled = compile_sim_death_count_field(
        normalized,
        registry_hashes=registry_set.hashes(),
        geography=geography,
    )

    field = write_compiled_field(run_dir, compiled)
    bundle.validate_minimal()

    return field.id