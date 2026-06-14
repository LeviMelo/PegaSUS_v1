from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.registries.semantic import active_entries


REGISTRY_FILE = "icd_catalog.yaml"


def icd_catalog_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    return active_entries(REGISTRY_FILE, registry_root=registry_root)


def chapter_for_code(code: str, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    normalized = str(code).upper().replace(".", "")[:3]
    if not normalized:
        return None
    for entry in icd_catalog_entries(registry_root=registry_root):
        bounds = entry.get("range") or []
        if len(bounds) != 2:
            continue
        lo, hi = str(bounds[0]).upper(), str(bounds[1]).upper()
        if lo <= normalized <= hi:
            return entry
    return None
