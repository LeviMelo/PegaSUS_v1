from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=256)
def _load_registry_file_cached(resolved: str, mtime: float) -> dict[str, Any]:
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Registry is not a mapping: {resolved}")
    return data


def load_registry_file(path: str | Path) -> dict[str, Any]:
    # Registry files are static within a run but this was re-parsed from disk on every
    # semantic.registry_entries / active_entries call (once per field during REG-07
    # classification). Cache the parse per (path, mtime) so edits still invalidate.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")
    return _load_registry_file_cached(str(path.resolve()), path.stat().st_mtime)
