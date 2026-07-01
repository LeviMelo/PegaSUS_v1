from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pegasus.sidra.schemas import SIDRARequest


def load_sidra_view_registry(path: str | Path = "config/registries/sidra/sidra_views.yaml") -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = payload.get("entries", [])
    return {str(entry["id"]): entry for entry in entries if "id" in entry}


def request_from_view(view_id: str, *, registry_path: str | Path = "config/registries/sidra/sidra_views.yaml") -> SIDRARequest:
    registry = load_sidra_view_registry(registry_path)
    if view_id not in registry:
        raise KeyError(f"SIDRA view not found in registry: {view_id}")

    entry = registry[view_id]
    return SIDRARequest(
        table_id=str(entry["table_id"]),
        variables=[str(x) for x in entry.get("variables", [])],
        periods=[str(x) for x in entry.get("periods", [])],
        locality_level=str(entry["locality_level"]),
        localities=[str(x) for x in entry.get("localities", [])],
        classifications={str(k): [str(x) for x in v] for k, v in entry.get("classifications", {}).items()},
    )
