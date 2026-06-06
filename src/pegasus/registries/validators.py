from __future__ import annotations

import json
from pathlib import Path

import yaml


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


def validate_registry_tree(root: str | Path = "config/registries") -> list[str]:
    root = Path(root)
    errors: list[str] = []
    required = [
        "registry_manifest.yaml",
        "source_fields.yaml",
        "composite_decoders.yaml",
        "carrier_registry.yaml",
        "unit_registry.yaml",
        "aggregation_registry.yaml",
        "provenance_registry.yaml",
        "quality_permissions.yaml",
        "race_axis_registry.yaml",
        "race_bridge_priors.yaml",
        "icd_catalog.yaml",
        "icd_quality_groups.yaml",
        "diagnostic_topology.yaml",
        "clinical_event_definitions.yaml",
        "cnes_capacity_registry.yaml",
        "sih_cost_registry.yaml",
        "sidra_table_seed.jsonl",
        "sidra_views.yaml",
        "sidra_category_maps.yaml",
        "sidra_stitching.yaml",
        "sidra_regime_registry.yaml",
        "municipality_crosswalk_sources.yaml",
        "join_affordances.yaml",
        "bridge_grammars.yaml",
        "stdfm_registry.yaml",
        "population_solver_registry.yaml",
        "model_registry.yaml",
        "residual_registry.yaml",
        "hsic_registry.yaml",
        "null_registry.yaml",
        "output_schema.yaml",
    ]
    for name in required:
        path = root / name
        if not path.exists():
            errors.append(f"missing registry: {name}")
        else:
            errors.extend(validate_registry_file(path))
    return errors
