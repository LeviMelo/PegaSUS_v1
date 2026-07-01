from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash


class ProvenanceRegistryError(ValueError):
    """Raised when provenance tags cannot be resolved."""


@dataclass(frozen=True)
class ProvenanceSpec:
    tag: str
    risk: float
    description: str
    registry_hash: str

    def as_manifest(self) -> dict[str, Any]:
        return {"tag": self.tag, "risk": self.risk, "description": self.description, "registry_hash": self.registry_hash}


def load_provenance_registry(registry_root: str | Path = "config/registries") -> dict[str, ProvenanceSpec]:
    path = Path(registry_root) / "ontology/provenance.yaml"
    payload = load_yaml(path)
    raw = payload.get("provenance_tags", {})
    if not isinstance(raw, dict) or not raw:
        raise ProvenanceRegistryError(f"provenance registry is empty or invalid: {path}")
    registry_hash = content_hash(payload)
    out: dict[str, ProvenanceSpec] = {}
    for tag, spec in raw.items():
        if not isinstance(spec, dict):
            raise ProvenanceRegistryError(f"provenance spec must be a mapping: {tag}")
        out[str(tag)] = ProvenanceSpec(
            tag=str(tag),
            risk=float(spec.get("risk", 1.0)),
            description=str(spec.get("description", "")),
            registry_hash=registry_hash,
        )
    return out


def get_provenance(tag: str, *, registry_root: str | Path = "config/registries") -> ProvenanceSpec:
    registry = load_provenance_registry(registry_root)
    try:
        return registry[tag]
    except KeyError as exc:
        raise ProvenanceRegistryError(f"unknown provenance tag: {tag}") from exc


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_provenance_registry(registry_root)
    return {"provenance_tags": {k: v.as_manifest() for k, v in sorted(registry.items())}}
