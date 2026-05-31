from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from pegasus.core.exceptions import RegistryError
from pegasus.core.hashing import sha256_file
from pegasus.registries.schemas import (
    AggregationRegistry,
    CarrierRegistry,
    GenericRegistry,
    ProvenanceRegistry,
    QualityPermissionsRegistry,
    RaceAxisRegistry,
    RegistryManifest,
    SourceFieldsRegistry,
    UnitRegistry,
)


REGISTRY_MODELS: dict[str, type[BaseModel]] = {
    "source_fields": SourceFieldsRegistry,
    "carrier_registry": CarrierRegistry,
    "unit_registry": UnitRegistry,
    "aggregation_registry": AggregationRegistry,
    "provenance_registry": ProvenanceRegistry,
    "quality_permissions": QualityPermissionsRegistry,
    "race_axis_registry": RaceAxisRegistry,
}


@dataclass(frozen=True)
class LoadedRegistry:
    name: str
    path: Path
    sha256: str
    payload: BaseModel


@dataclass(frozen=True)
class LoadedRegistrySet:
    manifest: RegistryManifest
    manifest_path: Path
    manifest_sha256: str
    registries: dict[str, LoadedRegistry]

    def hashes(self) -> dict[str, str]:
        out = {"registry_manifest": self.manifest_sha256}
        out.update({name: item.sha256 for name, item in self.registries.items()})
        return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"Registry file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise RegistryError(f"Registry file is empty: {path}")

    if not isinstance(data, dict):
        raise RegistryError(f"Registry file must contain a mapping: {path}")

    return data


def load_registry_manifest(registry_dir: str | Path) -> RegistryManifest:
    registry_dir = Path(registry_dir)
    manifest_path = registry_dir / "registry_manifest.yaml"
    return RegistryManifest.model_validate(read_yaml(manifest_path))


def load_registry(name: str, registry_dir: str | Path, relative_path: str) -> LoadedRegistry:
    registry_dir = Path(registry_dir)
    path = registry_dir / relative_path
    data = read_yaml(path)

    model_cls = REGISTRY_MODELS.get(name, GenericRegistry)

    try:
        payload = model_cls.model_validate(data)
    except Exception as exc:
        raise RegistryError(f"Failed to validate registry '{name}' at {path}: {exc}") from exc

    return LoadedRegistry(
        name=name,
        path=path,
        sha256=sha256_file(path),
        payload=payload,
    )


def load_registry_set(registry_dir: str | Path) -> LoadedRegistrySet:
    registry_dir = Path(registry_dir)
    manifest_path = registry_dir / "registry_manifest.yaml"
    manifest = load_registry_manifest(registry_dir)

    registries: dict[str, LoadedRegistry] = {}

    for name, item in manifest.registries.items():
        registries[name] = load_registry(name, registry_dir, item.path)

    return LoadedRegistrySet(
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        registries=registries,
    )