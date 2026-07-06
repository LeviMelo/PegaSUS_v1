"""Generic YAML registry loader for thin registry wrapper modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from pegasus.core.exceptions import RegistryValidationError
from pegasus.core.hashing import sha256_file


# The single §II.1 registry admission rule, shared by EVERY accessor stack. Previously
# generic.active_entries used a strict ``status == "active"`` test while semantic.active_entries
# used this graded-active union, so a naive loader unification to the strict form would silently
# drop graded-active entries — e.g. bridge_grammars' ``active_artifact_required`` / ``active_warning``
# EFG bridge-grammar operators. The two filters happen to agree on today's generic-fed data (all
# bare "active"), so no test caught the divergence: a latent gun. ``active*`` covers the graded
# variants (active_warning/active_reduced/active_observer/active_artifact_required/
# active_approximation/active_sparse/active_small_scale/active_gpu/...); stable/planned_contract/
# experimental are non-graded actives; deferred/legacy_identity/baseline/deprecated are NOT active.
_ACTIVE_NON_PREFIX_STATUSES: frozenset[str] = frozenset({"stable", "planned_contract", "experimental"})


def is_active(status: object) -> bool:
    """Whether a registry-entry status counts as active for admission (the single §II.1 rule)."""
    text = str(status or "")
    return text.startswith("active") or text in _ACTIVE_NON_PREFIX_STATUSES


@dataclass(frozen=True)
class RegistryEntry:
    id: str
    status: str
    description: str
    warnings: tuple[str, ...]
    payload: Mapping[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """dict-compatible read over the entry payload — lets consumers that were written against
        the semantic (``list[dict]``) contract treat a ``RegistryEntry`` uniformly (REG-07)."""
        return self.payload.get(key, default)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "description": self.description,
            "warnings": list(self.warnings),
            "payload": dict(self.payload),
        }


def _candidate_paths(registry_file: str | Sequence[str], root: str | Path) -> list[Path]:
    names = [registry_file] if isinstance(registry_file, str) else list(registry_file)
    return [Path(root) / name for name in names]


def resolve_registry_path(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> Path | None:
    candidates = _candidate_paths(registry_file, root)
    for path in candidates:
        if path.exists():
            return path
    if required:
        names = ", ".join(str(path) for path in candidates)
        raise RegistryValidationError(f"No registry file found among: {names}")
    return None


def load_registry_payload(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> dict[str, Any]:
    path = resolve_registry_path(registry_file, root=root, required=required)
    if path is None:
        return {"entries": [], "registry_file": None, "registry_missing": True}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RegistryValidationError(f"Registry payload must be a mapping: {path}")
    payload.setdefault("entries", [])
    payload["registry_file"] = str(path)
    payload["registry_sha256"] = sha256_file(path)
    return payload


def _entry(entry: Mapping[str, Any], index: int) -> RegistryEntry:
    entry_id = str(entry.get("id") or entry.get("name") or f"entry_{index}")
    warnings = entry.get("warnings") or ()
    if isinstance(warnings, str):
        warnings = (warnings,)
    return RegistryEntry(
        id=entry_id,
        status=str(entry.get("status") or "active"),
        description=str(entry.get("description") or entry_id),
        warnings=tuple(str(value) for value in warnings),
        payload=dict(entry),
    )


def load_entries(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> tuple[RegistryEntry, ...]:
    payload = load_registry_payload(registry_file, root=root, required=required)
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        raise RegistryValidationError("Registry entries must be a list")
    return tuple(_entry(entry, index) for index, entry in enumerate(entries) if isinstance(entry, Mapping))


def active_entries(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> tuple[RegistryEntry, ...]:
    return tuple(entry for entry in load_entries(registry_file, root=root, required=required) if is_active(entry.status))


def get_entry(
    registry_file: str | Sequence[str],
    entry_id: str,
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> RegistryEntry:
    for entry in load_entries(registry_file, root=root, required=required):
        if entry.id == entry_id:
            return entry
    raise RegistryValidationError(f"Entry not found in registry {registry_file}: {entry_id}")


def registry_manifest(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> dict[str, Any]:
    payload = load_registry_payload(registry_file, root=root, required=required)
    return {
        "registry_file": payload.get("registry_file"),
        "schema_version": payload.get("schema_version"),
        "registry_version": payload.get("registry_version"),
        "entry_count": len(payload.get("entries") or []),
        "registry_sha256": payload.get("registry_sha256"),
        "registry_missing": bool(payload.get("registry_missing")),
    }


__all__ = [
    "RegistryEntry",
    "is_active",
    "active_entries",
    "get_entry",
    "load_entries",
    "load_registry_payload",
    "registry_manifest",
    "resolve_registry_path",
]
