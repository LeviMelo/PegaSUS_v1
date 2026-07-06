from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class CarrierRegistryError(ValueError):
    """Raised when a carrier request cannot be resolved by the carrier registry."""


@dataclass(frozen=True)
class CarrierSpec:
    carrier_id: str
    label: str
    description: str
    allowed_units: tuple[str, ...]
    default_aggregation: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "carrier_id": self.carrier_id,
            "label": self.label,
            "description": self.description,
            "allowed_units": list(self.allowed_units),
            "default_aggregation": self.default_aggregation,
            "registry_hash": self.registry_hash,
        }


def load_carrier_registry(registry_root: str | Path = "config/registries") -> dict[str, CarrierSpec]:
    # Cache the built spec dict (and its registry_hash) per (root, mtime): get_carrier is
    # called once per source-field entry during registry validation, and previously each
    # call rebuilt every CarrierSpec and re-hashed the whole payload. mtime keys invalidation.
    path = Path(registry_root) / "ontology/carrier.yaml"
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return _load_carrier_registry_cached(str(Path(registry_root)), mtime)


@lru_cache(maxsize=16)
def _load_carrier_registry_cached(registry_root: str, mtime: float) -> dict[str, CarrierSpec]:
    path = Path(registry_root) / "ontology/carrier.yaml"
    payload = load_yaml(path)
    raw = payload.get("carriers", {})
    if not isinstance(raw, dict) or not raw:
        raise CarrierRegistryError(f"carrier registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, CarrierSpec] = {}
    for carrier_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise CarrierRegistryError(f"carrier spec must be a mapping: {carrier_id}")
        out[str(carrier_id)] = CarrierSpec(
            carrier_id=str(carrier_id),
            label=str(spec.get("label", carrier_id)),
            description=str(spec.get("description", "")),
            allowed_units=tuple(str(x) for x in spec.get("allowed_units", [])),
            default_aggregation=str(spec.get("default_aggregation", "non_aggregable")),
            registry_hash=registry_hash,
        )
    return out


def get_carrier(carrier_id: str, *, registry_root: str | Path = "config/registries") -> CarrierSpec:
    registry = load_carrier_registry(registry_root)
    try:
        return registry[carrier_id]
    except KeyError as exc:
        raise CarrierRegistryError(f"unknown carrier: {carrier_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_carrier_registry(registry_root)
    return {"carriers": {k: v.as_manifest() for k, v in sorted(registry.items())}}
