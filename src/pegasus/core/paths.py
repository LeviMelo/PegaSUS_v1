from __future__ import annotations

from pathlib import Path

from pegasus.core.config import load_yaml


DATA_LAKE_DIRS = [
    "data/raw/datasus/SIM-DO",
    "data/raw/datasus/SIH-RD",
    "data/raw/datasus/SINASC",
    "data/raw/datasus/CNES-ST",
    "data/raw/sidra/chunks",
    "data/raw/geo",
    "data/raw/external",
    "data/processed/datasus/SIM-DO",
    "data/processed/datasus/SIH-RD",
    "data/processed/datasus/SINASC",
    "data/processed/datasus/CNES-ST",
    "data/processed/sidra/facts",
    "data/processed/geo",
    "data/metadata/sidra/raw",
    "data/metadata/sidra/normalized",
    "data/metadata/datasus/profiles",
    "data/metadata/datasus/schema_compare",
    "data/metadata/datasus/variable_catalog",
    "data/metadata/registries",
    "data/metadata/geo",
    "data/cache/sidra/http",
    "data/cache/sidra/values",
    "data/cache/datasus/microdatasus",
    "data/manifests/datasus",
    "data/manifests/sidra",
    "data/manifests/runs",
    "data/intermediate/she",
    "data/intermediate/efg",
    "data/intermediate/pirs",
    "data/runs",
    "data/diagnostics",
]


def ensure_data_lake(root: str | Path = ".") -> list[Path]:
    root = Path(root)
    created = []
    for rel in DATA_LAKE_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        keep.touch(exist_ok=True)
        created.append(path)
    return created


def configured_paths(root: str | Path = ".") -> dict:
    return load_yaml(Path(root) / "config" / "paths.yaml")["paths"]
