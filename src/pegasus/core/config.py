from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pegasus.core.exceptions import ConfigError


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
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file is not a mapping: {path}")
    return data


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
