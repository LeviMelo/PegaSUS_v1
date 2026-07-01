from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class AggregationRegistryError(ValueError):
    """Raised when an aggregation law is missing or invalid."""


@dataclass(frozen=True)
class AggregationSpec:
    aggregation_id: str
    description: str
    allowed_for_rates_numerator: bool
    allowed_for_denominator: bool
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "aggregation_id": self.aggregation_id,
            "description": self.description,
            "allowed_for_rates_numerator": self.allowed_for_rates_numerator,
            "allowed_for_denominator": self.allowed_for_denominator,
            "registry_hash": self.registry_hash,
        }


def load_aggregation_registry(registry_root: str | Path = "config/registries") -> dict[str, AggregationSpec]:
    path = Path(registry_root) / "ontology/aggregation.yaml"
    payload = load_yaml(path)
    raw = payload.get("aggregations", {})
    if not isinstance(raw, dict) or not raw:
        raise AggregationRegistryError(f"aggregation registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, AggregationSpec] = {}
    for aggregation_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise AggregationRegistryError(f"aggregation spec must be a mapping: {aggregation_id}")
        out[str(aggregation_id)] = AggregationSpec(
            aggregation_id=str(aggregation_id),
            description=str(spec.get("description", "")),
            allowed_for_rates_numerator=bool(spec.get("allowed_for_rates_numerator", False)),
            allowed_for_denominator=bool(spec.get("allowed_for_denominator", False)),
            registry_hash=registry_hash,
        )
    return out


def get_aggregation(aggregation_id: str, *, registry_root: str | Path = "config/registries") -> AggregationSpec:
    registry = load_aggregation_registry(registry_root)
    try:
        return registry[aggregation_id]
    except KeyError as exc:
        raise AggregationRegistryError(f"unknown aggregation law: {aggregation_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_aggregation_registry(registry_root)
    return {"aggregations": {k: v.as_manifest() for k, v in sorted(registry.items())}}
