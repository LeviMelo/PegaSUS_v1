from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pegasus.core.exceptions import ConfigError


@lru_cache(maxsize=512)
def _load_yaml_cached(resolved: str, mtime: float) -> dict[str, Any]:
    """Parse a YAML config once per (path, mtime). Registry/config files are
    static within a run and were previously re-read from disk on every call —
    e.g. ``resolve_source_field`` re-parsed the source registry for every column
    of every record (~5e5 disk reads on a single municipality), which made SHE
    normalization effectively hang. Keying on mtime invalidates on edit."""
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file is not a mapping: {resolved}")
    return data


REQUIRED_CONFIGS = [
    "project.yaml",
    "paths.yaml",
    "compute.yaml",
    "datasus.yaml",
    "sidra.yaml",
    "output.yaml",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    return _load_yaml_cached(str(path.resolve()), path.stat().st_mtime)


def validate_config_tree(root: str | Path = ".") -> list[str]:
    root = Path(root)
    errors: list[str] = []
    for name in REQUIRED_CONFIGS:
        path = root / "config" / name
        if not path.exists():
            errors.append(f"missing config/{name}")
            continue
        try:
            load_yaml(path)
        except Exception as exc:
            errors.append(f"invalid config/{name}: {exc}")
    return errors


def load_all_configs(root: str | Path = ".") -> dict[str, dict[str, Any]]:
    root = Path(root)
    return {name: load_yaml(root / "config" / name) for name in REQUIRED_CONFIGS}
