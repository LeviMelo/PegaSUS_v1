from __future__ import annotations

from pathlib import Path

from pegasus.output.bundle import OutputBundle
from pegasus.output.field_store import get_field_row, read_field_data
from pegasus.output.write_tables import write_compiled_field
from pegasus.problem1.contracts import FieldNode, Lineage
from pegasus.problem1.rate_fields import compile_crude_mortality_rate_field
from pegasus.registries.index import RegistryIndex
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set


def _field_node_from_v_fields_row(row: dict) -> FieldNode:
    import json

    lineage = Lineage(
        parent_ids=[],
        operator_type=row["operator"] or "unknown",
        operator_params={},
        registry_versions={},
    )

    # Only the metadata required by downstream compilation is reconstructed here.
    # Full lineage comes from E_DAG and registry manifests; for local workflow
    # composition, the parent field ID and core semantic metadata are enough.
    return FieldNode(
        id=row["field_id"],
        name=row["name"],
        kind=row["kind"],
        carrier=row["carrier"],
        unit=row["unit"],
        support=json.loads(row["support_json"]),
        axes=json.loads(row["axes_json"]),
        aggregation=row["aggregation"],
        role=list(row["role"]),
        source=list(row["source"]),
        operator=row["operator"],
        provenance=list(row["provenance"]),
        state=row["state"],
        dashboard_safe=row["dashboard_safe"],
        warnings=list(row["warnings"]),
        lineage=lineage,
        materialization_state=row["materialization_state"],
        path=row["path"],
    )


def compile_crude_mortality_to_bundle(
    *,
    run_dir: str | Path,
    registry_dir: str | Path,
    death_count_field_id: str,
    population_field_id: str,
    scale: float = 100_000.0,
) -> str:
    run_dir = Path(run_dir)
    registry_dir = Path(registry_dir)

    bundle = OutputBundle(run_dir)
    bundle.validate_minimal()

    registry_set = load_registry_set(registry_dir)
    validate_registry_set(registry_set)
    registry_index = RegistryIndex.from_registry_set(registry_set)

    death_row = get_field_row(run_dir, death_count_field_id)
    population_row = get_field_row(run_dir, population_field_id)

    death_field = _field_node_from_v_fields_row(death_row)
    population_field = _field_node_from_v_fields_row(population_row)

    death_data = read_field_data(run_dir, death_count_field_id)
    population_data = read_field_data(run_dir, population_field_id)

    compiled = compile_crude_mortality_rate_field(
        death_count_field=death_field,
        death_count_data=death_data,
        population_field=population_field,
        population_data=population_data,
        registry_index=registry_index,
        registry_hashes=registry_set.hashes(),
        scale=scale,
    )

    field = write_compiled_field(run_dir, compiled)
    bundle.validate_minimal()

    return field.id