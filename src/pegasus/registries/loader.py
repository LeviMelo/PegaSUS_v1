from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pegasus.core.hashing import sha256_file


@dataclass(frozen=True)
class RegistryBundle:
    root: Path
    registries: dict[str, dict[str, Any]]
    hashes: dict[str, str]


def load_registry_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Registry is not a mapping: {path}")
    return data


def load_registries(root: str | Path = "config/registries") -> RegistryBundle:
    root = Path(root)
    registries: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    # Recursive: registries may live in a domain subdirectory (e.g. sidra/) rather
    # than flat under root.
    for path in sorted(root.rglob("*.yaml")):
        registries[path.stem] = load_registry_file(path)
        hashes[path.stem] = sha256_file(path)
    seed = root / "sidra" / "sidra_table_seed.jsonl"
    if seed.exists():
        hashes["sidra_table_seed"] = sha256_file(seed)
    return RegistryBundle(root=root, registries=registries, hashes=hashes)
