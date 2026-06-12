from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class UnitRegistryError(ValueError):
    """Raised when a unit request cannot be resolved by the unit registry."""


@dataclass(frozen=True)
class UnitSpec:
    unit_id: str
    dimension: str
    additive: bool
    description: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "dimension": self.dimension,
            "additive": self.additive,
            "description": self.description,
            "registry_hash": self.registry_hash,
        }


def load_unit_registry(registry_root: str | Path = "config/registries") -> dict[str, UnitSpec]:
    path = Path(registry_root) / "unit.yaml"
    payload = load_yaml(path)
    raw = payload.get("units", {})
    if not isinstance(raw, dict) or not raw:
        raise UnitRegistryError(f"unit registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, UnitSpec] = {}
    for unit_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise UnitRegistryError(f"unit spec must be a mapping: {unit_id}")
        out[str(unit_id)] = UnitSpec(
            unit_id=str(unit_id),
            dimension=str(spec.get("dimension", "unknown")),
            additive=bool(spec.get("additive", False)),
            description=str(spec.get("description", "")),
            registry_hash=registry_hash,
        )
    return out


def get_unit(unit_id: str, *, registry_root: str | Path = "config/registries") -> UnitSpec:
    registry = load_unit_registry(registry_root)
    try:
        return registry[unit_id]
    except KeyError as exc:
        raise UnitRegistryError(f"unknown unit: {unit_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_unit_registry(registry_root)
    return {"units": {k: v.as_manifest() for k, v in sorted(registry.items())}}
