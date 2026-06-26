"""MSD §3.10 Core Seed Registry — bind canonical seed IDs to produced EFG fields.

The MSD prescribes a *named* core-seed surface (V_M01 CrudeMortality, V_C01
InfantMortality, …) and intents declare ``mandatory_fields`` against those names. The
autonomous EFG produces content-addressed (hashed) field ids, so the names never matched
and ``mandatory_fields`` could not be enforced — the "hollow success" the audit flagged.

This module resolves the gap *without* renaming content-addressed fields: a declarative
match predicate (``config/registries/core_seed_registry.yaml``) binds each canonical
seed id/name/alias to the produced field that realizes it. ``mandatory_fields`` are then
enforced against real output. Registry-driven: adding a seed is a registry edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pegasus.registries.generic import load_entries


REGISTRY_FILE = "core_seed_registry.yaml"


class MandatoryFieldContractError(ValueError):
    """Raised when an intent's ``mandatory_fields`` are not produced by the compiled EFG."""


@dataclass(frozen=True)
class CoreSeedSpec:
    seed_id: str
    name: str
    aliases: tuple[str, ...]
    match: dict[str, Any]

    def identifiers(self) -> set[str]:
        return {self.seed_id, self.name, *self.aliases}


@lru_cache(maxsize=8)
def _seed_specs_cached(root: str) -> tuple[CoreSeedSpec, ...]:
    specs: list[CoreSeedSpec] = []
    for entry in load_entries(REGISTRY_FILE, root=root, required=False):
        payload = entry.payload
        match = payload.get("match")
        if not isinstance(match, dict):
            continue
        specs.append(CoreSeedSpec(
            seed_id=str(payload.get("id") or entry.id),
            name=str(payload.get("name") or payload.get("id") or entry.id),
            aliases=tuple(str(a) for a in (payload.get("aliases") or [])),
            match=dict(match),
        ))
    return tuple(specs)


def core_seed_specs(*, registry_root: str | Path = "config/registries") -> tuple[CoreSeedSpec, ...]:
    return _seed_specs_cached(str(registry_root))


def _field_attr(field: Any, name: str, default: Any = None) -> Any:
    if isinstance(field, dict):
        return field.get(name, default)
    return getattr(field, name, default)


def _field_matches(field: Any, match: dict[str, Any]) -> bool:
    roles = {str(r) for r in (_field_attr(field, "role", []) or [])}
    axes = dict(_field_attr(field, "axes", {}) or {})
    support = dict(_field_attr(field, "support", {}) or {})
    carrier = str(_field_attr(field, "carrier", ""))

    if match.get("population"):
        return "population_tensor" in roles
    role = match.get("role")
    if role is not None and str(role) not in roles:
        return False
    want_carrier = match.get("carrier")
    if want_carrier is not None and carrier != str(want_carrier):
        return False
    requires_axis = match.get("requires_axis")
    if requires_axis is not None and str(requires_axis) not in axes:
        return False
    for axis in match.get("excludes_axes", []) or []:
        if str(axis) in axes:
            return False
    restrict = match.get("restrict_predicate")
    if restrict is not None and str(support.get("restrict_predicate") or "") != str(restrict):
        return False
    return True


def resolve_core_seeds(
    fields: Iterable[Any], *, registry_root: str | Path = "config/registries"
) -> dict[str, str]:
    """Map each canonical seed id to the field id that realizes it (first match wins)."""
    field_list = list(fields)
    resolved: dict[str, str] = {}
    for spec in core_seed_specs(registry_root=registry_root):
        for field in field_list:
            if _field_matches(field, spec.match):
                resolved[spec.seed_id] = str(_field_attr(field, "id", "") or _field_attr(field, "field_id", ""))
                break
    return resolved


def _intent_mandatory_fields(intent: Any) -> list[str]:
    if intent is None:
        return []
    value = intent.get("mandatory_fields", []) if isinstance(intent, dict) else getattr(intent, "mandatory_fields", [])
    return [str(v) for v in (value or [])]


def enforce_mandatory_fields(
    intent: Any, fields: Iterable[Any], *, registry_root: str | Path = "config/registries"
) -> dict[str, Any]:
    """Fail the compile if an intent's ``mandatory_fields`` are not produced (MSD §3.10).

    A mandatory token resolves when it matches a core-seed id/name/alias *and* that seed
    binds to a produced field. Unknown tokens (not in the registry at all) are reported
    as unresolved rather than silently passing.
    """
    mandatory = _intent_mandatory_fields(intent)
    if not mandatory:
        return {"requested": [], "satisfied": [], "unsatisfied": [], "seed_field_map": {}}

    specs = core_seed_specs(registry_root=registry_root)
    resolved = resolve_core_seeds(fields, registry_root=registry_root)
    # token -> seed_id (via id/name/alias)
    token_to_seed: dict[str, str] = {}
    for spec in specs:
        for token in spec.identifiers():
            token_to_seed[token] = spec.seed_id

    satisfied: list[str] = []
    unsatisfied: list[str] = []
    seed_field_map: dict[str, str] = {}
    for token in mandatory:
        seed_id = token_to_seed.get(token)
        if seed_id is not None and seed_id in resolved:
            satisfied.append(token)
            seed_field_map[token] = resolved[seed_id]
        else:
            unsatisfied.append(token)

    summary = {
        "requested": mandatory,
        "satisfied": sorted(satisfied),
        "unsatisfied": sorted(unsatisfied),
        "seed_field_map": seed_field_map,
        "resolved_seed_count": len(resolved),
    }
    if unsatisfied:
        raise MandatoryFieldContractError(
            f"EFG did not produce intent mandatory_fields {sorted(unsatisfied)}; "
            f"resolved core seeds: {sorted(resolved)}. Refusing a hollow compile success."
        )
    return summary


__all__ = [
    "MandatoryFieldContractError",
    "CoreSeedSpec",
    "core_seed_specs",
    "resolve_core_seeds",
    "enforce_mandatory_fields",
]
