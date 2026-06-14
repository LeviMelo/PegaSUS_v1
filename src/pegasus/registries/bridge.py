from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.registries.semantic import active_entries


REGISTRY_FILE = "bridge_grammars.yaml"


def bridge_grammar_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    return active_entries(REGISTRY_FILE, registry_root=registry_root)


def bridge_grammar_for_type(bridge_type: str, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    for entry in bridge_grammar_entries(registry_root=registry_root):
        if entry.get("bridge_type") == bridge_type or entry.get("id") == bridge_type:
            return entry
    return None


def bridge_operator_is_registered(operator: str, *, registry_root: str | Path = "config/registries") -> bool:
    for entry in bridge_grammar_entries(registry_root=registry_root):
        if operator in set(entry.get("operators", []) or []):
            return True
    return False
