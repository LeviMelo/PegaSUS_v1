"""Thin registry wrapper for clinical event registry entries.

This module is the read boundary for ``health/clinical_event_definitions.yaml`` — the
SINGLE SOURCE OF TRUTH for clinical event carriers and their legal RN ratios
(MSD §2.6, §3.10.4).  The EFG engine consults the helpers here instead of
hardcoding carrier/ratio knowledge, so adding a data source or a derived event
is a registry edit, not an engine edit (source-agnostic principle, §1.1/§2.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pegasus.registries.generic import RegistryEntry, active_entries, get_entry, load_entries, registry_manifest

REGISTRY_FILES = ('health/clinical_event_definitions.yaml', 'events_registry.yaml', 'events.yaml')

# Denominator-unit policy by ratio output kind. A person-time denominator yields a
# rate; an event-count denominator yields a proportion. This is a general
# epidemiological rule (not source-specific) and the only unit logic the engine
# needs beyond the registry-declared (carrier, carrier, role) pairings.
_DENOMINATOR_UNITS_BY_OUTPUT: dict[str, tuple[str, ...]] = {
    "rate": ("person_years", "persons"),
    "proportion": ("counts",),
}


@dataclass(frozen=True)
class RestrictedEventSpec:
    """A σ-restricted clinical event (MSD §2.6/§3.10.4): a declarative predicate over a
    primary carrier's source records that yields a derived event carrier."""

    event_id: str
    event_carrier: str
    of_carrier: str
    predicate: str
    conditions: tuple[dict, ...]

    def has_conditions(self) -> bool:
        return bool(self.conditions)


@dataclass(frozen=True)
class ClinicalRatioSpec:
    """One legal Radon–Nikodym ratio declared by the clinical event registry."""

    numerator_carrier: str
    denominator_carrier: str
    role: str
    output_unit: str
    numerator_unit: str
    event_id: str

    @property
    def denominator_units(self) -> tuple[str, ...]:
        return _DENOMINATOR_UNITS_BY_OUTPUT.get(self.output_unit, ("counts",))


def load_event_entries(*, root: str | Path = "config/registries", required: bool = False) -> tuple[RegistryEntry, ...]:
    return load_entries(REGISTRY_FILES, root=root, required=required)


def active_event_entries(*, root: str | Path = "config/registries", required: bool = False) -> tuple[RegistryEntry, ...]:
    return active_entries(REGISTRY_FILES, root=root, required=required)


def get_event_entry(entry_id: str, *, root: str | Path = "config/registries", required: bool = True) -> RegistryEntry:
    return get_entry(REGISTRY_FILES, entry_id, root=root, required=required)


def event_registry_manifest(*, root: str | Path = "config/registries", required: bool = False) -> dict:
    return registry_manifest(REGISTRY_FILES, root=root, required=required)


@lru_cache(maxsize=8)
def _ratio_specs_cached(root: str) -> tuple[ClinicalRatioSpec, ...]:
    specs: list[ClinicalRatioSpec] = []
    for entry in active_entries(REGISTRY_FILES, root=root, required=False):
        payload = entry.payload
        carrier = payload.get("event_carrier")
        if not carrier:
            continue
        numerator_unit = str(payload.get("numerator_unit") or "counts")
        for ratio in payload.get("ratios") or []:
            if not isinstance(ratio, dict):
                continue
            denom = ratio.get("denominator_carrier")
            role = ratio.get("role")
            if not denom or not role:
                continue
            specs.append(ClinicalRatioSpec(
                numerator_carrier=str(carrier),
                denominator_carrier=str(denom),
                role=str(role),
                output_unit=str(ratio.get("output_unit") or "rate"),
                numerator_unit=numerator_unit,
                event_id=entry.id,
            ))
    return tuple(specs)


def clinical_ratio_specs(*, root: str | Path = "config/registries") -> tuple[ClinicalRatioSpec, ...]:
    """All legal RN ratios declared by the clinical event registry."""
    return _ratio_specs_cached(str(root))


@lru_cache(maxsize=8)
def _restricted_event_specs_cached(root: str) -> tuple[RestrictedEventSpec, ...]:
    specs: list[RestrictedEventSpec] = []
    for entry in active_entries(REGISTRY_FILES, root=root, required=False):
        payload = entry.payload
        if str(payload.get("event_kind") or "primary") != "restricted":
            continue
        carrier = payload.get("event_carrier")
        restriction = payload.get("restriction") or {}
        of_carrier = restriction.get("of_carrier")
        if not carrier or not of_carrier:
            continue
        conditions = tuple(
            dict(cond) for cond in (restriction.get("conditions") or []) if isinstance(cond, dict)
        )
        specs.append(RestrictedEventSpec(
            event_id=entry.id,
            event_carrier=str(carrier),
            of_carrier=str(of_carrier),
            predicate=str(restriction.get("predicate") or entry.id),
            conditions=conditions,
        ))
    return tuple(specs)


def restricted_event_specs(*, root: str | Path = "config/registries") -> tuple[RestrictedEventSpec, ...]:
    """All σ-restricted clinical events declared by the registry (infant death, LBW, …)."""
    return _restricted_event_specs_cached(str(root))


@lru_cache(maxsize=8)
def _primary_event_carriers_cached(root: str) -> frozenset[str]:
    carriers: set[str] = set()
    for entry in active_entries(REGISTRY_FILES, root=root, required=False):
        payload = entry.payload
        carrier = payload.get("event_carrier")
        if carrier and str(payload.get("event_kind") or "primary") == "primary":
            carriers.add(str(carrier))
    return frozenset(carriers)


def primary_event_carriers(*, root: str | Path = "config/registries") -> frozenset[str]:
    """Event carriers that are directly countable from a source artifact (COUNT_MEASURE)."""
    return _primary_event_carriers_cached(str(root))


@lru_cache(maxsize=8)
def _known_event_carriers_cached(root: str) -> frozenset[str]:
    carriers: set[str] = set()
    for entry in active_entries(REGISTRY_FILES, root=root, required=False):
        carrier = entry.payload.get("event_carrier")
        if carrier:
            carriers.add(str(carrier))
    return frozenset(carriers)


def known_event_carriers(*, root: str | Path = "config/registries") -> frozenset[str]:
    """All event carriers declared by the clinical event registry (primary + restricted)."""
    return _known_event_carriers_cached(str(root))


__all__ = [
    "REGISTRY_FILES",
    "ClinicalRatioSpec",
    "RestrictedEventSpec",
    "clinical_ratio_specs",
    "restricted_event_specs",
    "known_event_carriers",
    "primary_event_carriers",
    "load_event_entries",
    "active_event_entries",
    "get_event_entry",
    "event_registry_manifest",
]
