from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.registries.loader import load_registry_file


def registry_root_path(root: str | Path = "config/registries") -> Path:
    return Path(root)


def registry_entries(name: str, *, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    payload = load_registry_file(Path(registry_root) / name)
    entries = payload.get("entries", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def active_entries(name: str, *, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    entries = registry_entries(name, registry_root=registry_root)
    return [
        entry
        for entry in entries
        if str(entry.get("status", "")).startswith("active")
        or str(entry.get("status", "")) in {"stable", "planned_contract", "experimental"}
    ]


def field_text(field: Any) -> str:
    parts: list[str] = []
    for attr in (
        "field_id",
        "id",
        "name",
        "source",
        "source_json",
        "role",
        "role_json",
        "metadata_json",
        "operator",
        "carrier",
        "unit",
        "aggregation",
    ):
        value = getattr(field, attr, None)
        if value is not None:
            parts.append(str(value))
    try:
        dump = field.model_dump()
    except Exception:
        dump = None
    if isinstance(dump, dict):
        parts.extend(str(value) for value in dump.values() if value is not None)
    return " ".join(parts).upper()


def match_entry(field: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most specific registry entry matching a field descriptor.

    Slice 27A originally returned the first matching entry.  That was too broad:
    a generic CNES pattern matched before QTLEIT/QTINST capacity-vector patterns,
    causing QTLEIT fields to be classified as facility_stock.  Specificity is now
    defined by the longest matching pattern, with entry-id matches treated as high
    specificity.  This preserves deterministic behavior while avoiding generic
    registry entries shadowing specialized semantics.
    """
    haystack = field_text(field)
    best: tuple[int, int, dict[str, Any]] | None = None
    for order, entry in enumerate(entries):
        score = -1
        patterns = entry.get("field_patterns", []) or []
        for pattern in patterns:
            token = str(pattern).upper()
            if token and token in haystack:
                score = max(score, len(token))
        entry_id = str(entry.get("id", "")).upper()
        if entry_id and entry_id in haystack:
            score = max(score, len(entry_id) + 1000)
        if score >= 0 and (best is None or score > best[0] or (score == best[0] and order < best[1])):
            best = (score, order, entry)
    return None if best is None else best[2]


def registry_is_scaffold_only(name: str, *, registry_root: str | Path = "config/registries") -> bool:
    entries = registry_entries(name, registry_root=registry_root)
    if not entries:
        return True
    if len(entries) == 1:
        entry = entries[0]
        warnings = {str(value) for value in entry.get("warnings", []) or []}
        status = str(entry.get("status", ""))
        return status == "deferred" and "scaffold_only" in warnings
    return all("scaffold_only" in {str(value) for value in entry.get("warnings", []) or []} for entry in entries)
