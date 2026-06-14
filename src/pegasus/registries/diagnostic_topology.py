from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.registries.semantic import active_entries, match_entry


REGISTRY_FILE = "diagnostic_topology.yaml"


def diagnostic_topology_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    return active_entries(REGISTRY_FILE, registry_root=registry_root)


def diagnostic_topology_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    return match_entry(field, diagnostic_topology_entries(registry_root=registry_root))


def is_diagnostic_topology_field(field: Any, *, registry_root: str | Path = "config/registries") -> bool:
    return diagnostic_topology_for_field(field, registry_root=registry_root) is not None


def diagnostic_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    entry = diagnostic_topology_for_field(field, registry_root=registry_root)
    if entry is None:
        return None
    return {
        "registry": REGISTRY_FILE,
        "entry_id": entry.get("id"),
        "topology_role": entry.get("topology_role"),
        "unit": entry.get("unit"),
        "aggregation": entry.get("aggregation"),
        "forbidden_operators": list(entry.get("forbidden_operators", []) or []),
    }
