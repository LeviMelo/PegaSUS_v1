from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class QualityRegistryError(ValueError):
    """Raised when a quality role cannot be resolved."""


@dataclass(frozen=True)
class QualityRoleSpec:
    role_id: str
    substrate_admissible: bool
    default_state: str
    description: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "substrate_admissible": self.substrate_admissible,
            "default_state": self.default_state,
            "description": self.description,
            "registry_hash": self.registry_hash,
        }


def load_quality_registry(registry_root: str | Path = "config/registries") -> dict[str, QualityRoleSpec]:
    path = Path(registry_root) / "quality.yaml"
    payload = load_yaml(path)
    raw = payload.get("quality_roles", {})
    if not isinstance(raw, dict) or not raw:
        raise QualityRegistryError(f"quality registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, QualityRoleSpec] = {}
    for role_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise QualityRegistryError(f"quality role spec must be a mapping: {role_id}")
        out[str(role_id)] = QualityRoleSpec(
            role_id=str(role_id),
            substrate_admissible=bool(spec.get("substrate_admissible", False)),
            default_state=str(spec.get("default_state", "quarantined_descriptive")),
            description=str(spec.get("description", "")),
            registry_hash=registry_hash,
        )
    return out


def get_quality_role(role_id: str, *, registry_root: str | Path = "config/registries") -> QualityRoleSpec:
    registry = load_quality_registry(registry_root)
    try:
        return registry[role_id]
    except KeyError as exc:
        raise QualityRegistryError(f"unknown quality role: {role_id}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_quality_registry(registry_root)
    return {"quality_roles": {k: v.as_manifest() for k, v in sorted(registry.items())}}
