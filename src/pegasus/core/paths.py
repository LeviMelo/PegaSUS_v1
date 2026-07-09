from __future__ import annotations

from pathlib import Path

from pegasus.core.config import load_yaml
from pegasus.core.data_lifecycle import classify

# The per-role persistence law lives in pegasus.core.data_lifecycle (the single source of truth for
# what each data/ root IS and whether it may be reclaimed). DATA_LAKE_DIRS below is just the durable
# skeleton ensure_data_lake pre-creates; validate_lake_against_contract() keeps the two from drifting.

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
    "data/runs",
    "data/diagnostics",
]
# Note: data/intermediate/* is intentionally NOT pre-created — per the lifecycle contract it is an
# EPHEMERAL role that is never actually used (real per-run stage workspaces are written adjacent to
# the run_dir, not here).


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


def validate_lake_against_contract() -> list[str]:
    """Ensure every pre-created lake dir falls under a governed lifecycle root (no drift).

    Returns a list of DATA_LAKE_DIRS entries that no :data:`DATA_LIFECYCLE` root governs — empty when
    the skeleton and the contract agree.
    """
    return [rel for rel in DATA_LAKE_DIRS if classify(rel) is None]


def configured_paths(root: str | Path = ".") -> dict:
    return load_yaml(Path(root) / "config" / "paths.yaml")["paths"]
