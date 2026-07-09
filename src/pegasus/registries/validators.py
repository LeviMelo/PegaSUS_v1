from __future__ import annotations

import json
from pathlib import Path

import yaml

from pegasus.registries.callables import resolve_callable


REQUIRED_REGISTRY_KEYS = {
    "schema_version",
    "registry_version",
    "created_at",
    "updated_at",
    "provenance",
    "entries",
}


def validate_registry_file(path: str | Path) -> list[str]:
    path = Path(path)
    errors: list[str] = []

    if path.name in {"carrier.yaml", "unit.yaml", "aggregation.yaml", "provenance.yaml"}:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return [f"{path.name}: registry is not a mapping"]
        if "schema_version" not in data:
            errors.append(f"{path.name}: missing schema_version")
        if "registry_id" not in data:
            errors.append(f"{path.name}: missing registry_id")
        semantic_key = {
            "carrier.yaml": "carriers",
            "unit.yaml": "units",
            "aggregation.yaml": "aggregations",
            "provenance.yaml": "provenance_tags",
        }[path.name]
        values = data.get(semantic_key)
        if not isinstance(values, dict) or not values:
            errors.append(f"{path.name}: missing nonempty {semantic_key}")
        return errors

    if path.name == "registry_manifest.yaml":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "registry_set" not in data:
            errors.append("registry_manifest missing registry_set")
        if "registries" not in data:
            errors.append("registry_manifest missing registries")
        return errors

    if path.suffix == ".jsonl":
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{i}: invalid jsonl: {exc}")
        return errors

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        return [f"{path.name}: registry is not a mapping"]

    missing = REQUIRED_REGISTRY_KEYS - set(data)
    for key in sorted(missing):
        errors.append(f"{path.name}: missing {key}")

    if "entries" in data and not isinstance(data["entries"], list):
        errors.append(f"{path.name}: entries is not a list")

    for idx, entry in enumerate(data.get("entries", [])):
        if not isinstance(entry, dict):
            errors.append(f"{path.name}: entry {idx} is not a mapping")
            continue
        for key in ["id", "status", "description", "warnings"]:
            if key not in entry:
                errors.append(f"{path.name}: entry {idx} missing {key}")

    return errors


def _iter_source_field_specs(source_fields: dict):
    """Yield (source_system, canonical_field, spec) over datasus/source_fields.yaml."""
    for source_system, block in (source_fields.get("source_systems") or {}).items():
        if not isinstance(block, dict):
            continue
        for group in ("fields", "field_patterns"):
            for canonical, spec in (block.get(group) or {}).items():
                if isinstance(spec, dict):
                    yield source_system, canonical, spec


def validate_registry_authority(root: str | Path = "config/registries") -> list[str]:
    """Executable-authority cross-registry checks (MSD-II §II.1 / MII-REG-07).

    Beyond the file-presence/shape checks of ``validate_registry_tree``, this
    proves the registry is *executable*:

    1. every ``decoder``/``parser`` name declared in ``datasus/source_fields.yaml``
       resolves to a real callable through the single ``resolve_callable``
       resolver (no dangling callable references);
    2. every ``carrier``/``unit``/``aggregation`` token a source field declares
       exists in the respective vocabulary registry (no drift);
    3. round-trip routing: a field routed ``Decode``/``Parse`` must name the
       callable that produces it (no unroutable admissible field).
    """
    root = Path(root)
    errors: list[str] = []

    def _load(name: str) -> dict:
        path = root / name
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    source_fields = _load("datasus/source_fields.yaml")
    if not source_fields:
        return [f"datasus/source_fields.yaml missing or empty under {root}"]

    carriers = set((_load("ontology/carrier.yaml").get("carriers") or {}))
    units = set((_load("ontology/unit.yaml").get("units") or {}))
    aggregations = set((_load("ontology/aggregation.yaml").get("aggregations") or {}))

    for source_system, canonical, spec in _iter_source_field_specs(source_fields):
        where = f"datasus/source_fields.yaml[{source_system}.{canonical}]"

        for key in ("decoder", "parser", "composite_decoder", "transform"):
            name = spec.get(key)
            if name and resolve_callable(str(name)) is None:
                errors.append(f"{where}: {key} '{name}' does not resolve to a callable")

        # Vocabulary consistency (only when the registry declares that vocabulary).
        carrier = spec.get("carrier")
        if carrier and carriers and str(carrier) not in carriers:
            errors.append(f"{where}: carrier '{carrier}' not in ontology/carrier.yaml")
        unit = spec.get("unit")
        if unit and units and str(unit) not in units:
            errors.append(f"{where}: unit '{unit}' not in ontology/unit.yaml")
        aggregation = spec.get("aggregation")
        if aggregation and aggregations and str(aggregation) not in aggregations:
            errors.append(f"{where}: aggregation '{aggregation}' not in ontology/aggregation.yaml")

        # Round-trip routing: a Decode/Parse route must name its callable.
        route = str(spec.get("route") or "").lower()
        if route == "decode" and not (spec.get("decoder") or spec.get("composite_decoder") or spec.get("transform")):
            errors.append(f"{where}: route=Decode but no decoder is declared")
        if route == "parse" and not (spec.get("parser") or spec.get("transform")):
            errors.append(f"{where}: route=Parse but no parser is declared")

    # SpatialWeightGraph registry (MSD-II §II.4): every declared graph must carry
    # legality_class + provenance and point at an existing artifact.
    spatial = _load("spatial/spatial_graphs.yaml")
    for graph_id, spec in (spatial.get("graphs") or {}).items():
        if not isinstance(spec, dict):
            errors.append(f"spatial/spatial_graphs.yaml[{graph_id}]: graph spec is not a mapping")
            continue
        if spec.get("legality_class") not in {"structural", "context_derived"}:
            errors.append(f"spatial/spatial_graphs.yaml[{graph_id}]: legality_class must be structural|context_derived")
        if not spec.get("provenance"):
            errors.append(f"spatial/spatial_graphs.yaml[{graph_id}]: missing provenance")
        artifact = spec.get("artifact")
        if not artifact:
            errors.append(f"spatial/spatial_graphs.yaml[{graph_id}]: missing artifact")
        elif not (root / str(artifact)).exists():
            errors.append(f"spatial/spatial_graphs.yaml[{graph_id}]: artifact '{artifact}' does not exist")

    return errors


def validate_registry_tree(root: str | Path = "config/registries") -> list[str]:
    root = Path(root)
    errors: list[str] = []
    required = [
        "registry_manifest.yaml",
        "datasus/source_fields.yaml",
        "datasus/composite_decoders.yaml",
        "ontology/carrier.yaml",
        "ontology/unit.yaml",
        "ontology/aggregation.yaml",
        "ontology/provenance.yaml",
        "ontology/quality_permissions.yaml",
        "demographic/race_axis_registry.yaml",
        "demographic/race_bridge_priors.yaml",
        "health/icd_catalog.yaml",
        "health/icd_quality_groups.yaml",
        "health/diagnostic_topology.yaml",
        "health/clinical_event_definitions.yaml",
        "health/cnes_capacity_registry.yaml",
        "health/sih_cost_registry.yaml",
        "health/denominators.yaml",
        "sidra/sidra_table_seed.jsonl",
        "sidra/sidra_views.yaml",
        "sidra/sidra_category_maps.yaml",
        "sidra/sidra_stitching.yaml",
        "sidra/sidra_regime_registry.yaml",
        "spatial/municipality_crosswalk_sources.yaml",
        "fields/join_affordances.yaml",
        "fields/bridge_grammars.yaml",
        "inference/stdfm_registry.yaml",
        "demographic/population_solver_registry.yaml",
        "inference/model_registry.yaml",
        "inference/residual_registry.yaml",
        "inference/hsic_registry.yaml",
        "ontology/null_registry.yaml",
        "output_schema.yaml",
    ]
    for name in required:
        path = root / name
        if not path.exists():
            errors.append(f"missing registry: {name}")
        else:
            errors.extend(validate_registry_file(path))
    # Executable-authority cross-registry checks (MSD-II §II.1 / MII-REG-07):
    # file presence + shape is necessary but not sufficient; the registry must
    # also be executable (every declared callable resolves, vocabularies agree,
    # routes name their callable).
    errors.extend(validate_registry_authority(root))
    return errors
