"""IBGE geographic-hierarchy crosswalk: municipality (cod6) -> coarser region.

Backs the EFG spatial-aggregation level (MSD §3.7 geography). Aggregating events to
a coarser IBGE region (microregion / immediate region / mesoregion / intermediate
region) raises per-cell counts, which is the legitimate lever for analysing outcomes
that are too sparse at the municipality level. The crosswalk is authoritative IBGE
data (``config/registries/spatial/geography_region_crosswalk.yaml``), never derived from code
positions or fabricated.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

RegionLevel = Literal["microregion", "immediate_region", "mesoregion", "intermediate_region"]

_LEVEL_KEY = {
    "microregion": "microregion_id",
    "immediate_region": "immediate_region_id",
    "mesoregion": "mesoregion_id",
    "intermediate_region": "intermediate_region_id",
}

REGISTRY_FILE = "spatial/geography_region_crosswalk.yaml"


class RegionCrosswalkError(ValueError):
    """Raised when a requested aggregation level is unknown or the crosswalk is absent."""


@lru_cache(maxsize=8)
def _crosswalk(registry_root: str) -> dict[str, dict[str, str]]:
    """cod6 -> {level_key: region_id} from the IBGE hierarchy registry."""
    path = Path(registry_root) / REGISTRY_FILE
    if not path.exists():
        raise RegionCrosswalkError(f"geography region crosswalk not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, str]] = {}
    for entry in payload.get("entries", []) or []:
        cod6 = entry.get("cod6") or (str(entry.get("cod7"))[:6] if entry.get("cod7") else None)
        if not cod6:
            continue
        out[str(cod6)] = {
            key: str(entry[key])
            for key in _LEVEL_KEY.values()
            if entry.get(key) not in (None, "", "None")
        }
    return out


def region_for_cod6(cod6: str, level: RegionLevel, *, registry_root: str | Path = "config/registries") -> str | None:
    """Map a 6-digit municipality code to its IBGE region id at ``level`` (None if unmapped)."""
    if level not in _LEVEL_KEY:
        raise RegionCrosswalkError(f"unknown aggregation level: {level!r}")
    record = _crosswalk(str(registry_root)).get(str(cod6))
    if not record:
        return None
    return record.get(_LEVEL_KEY[level])


def cod6_to_region_map(level: RegionLevel, *, registry_root: str | Path = "config/registries") -> dict[str, str]:
    """Full cod6 -> region-id mapping at ``level`` (for vectorized joins)."""
    if level not in _LEVEL_KEY:
        raise RegionCrosswalkError(f"unknown aggregation level: {level!r}")
    key = _LEVEL_KEY[level]
    table = _crosswalk(str(registry_root))
    return {cod6: rec[key] for cod6, rec in table.items() if key in rec}


__all__ = ["RegionLevel", "RegionCrosswalkError", "region_for_cod6", "cod6_to_region_map"]
