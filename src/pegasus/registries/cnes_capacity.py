from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pegasus.registries.semantic import active_entries, match_entry


class CNESCapacityRegistryError(ValueError):
    """Raised when CNES capacity requests erase vector-indexed semantics."""


@dataclass(frozen=True)
class CNESCapacityComponent:
    raw_field: str
    component_id: str
    carrier: str
    unit: str
    family: Literal["bed", "room"]
    label: str


CAPACITY_COMPONENTS: dict[str, CNESCapacityComponent] = {
    "QTLEITP1": CNESCapacityComponent("QTLEITP1", "clinical_bed_capacity", "FacilityCapacityVector", "facility_capacity_units_QTLEITP1", "bed", "Clinical beds"),
    "QTLEITP2": CNESCapacityComponent("QTLEITP2", "surgical_bed_capacity", "FacilityCapacityVector", "facility_capacity_units_QTLEITP2", "bed", "Surgical beds"),
    "QTLEITP3": CNESCapacityComponent("QTLEITP3", "obstetric_bed_capacity", "FacilityCapacityVector", "facility_capacity_units_QTLEITP3", "bed", "Obstetric beds"),
    "QTINST01": CNESCapacityComponent("QTINST01", "consulting_room_capacity", "FacilityCapacityVector", "facility_capacity_units_QTINST01", "room", "Consulting rooms / infrastructure component 01"),
    "QTINST34": CNESCapacityComponent("QTINST34", "room_infrastructure_capacity_34", "FacilityCapacityVector", "facility_capacity_units_QTINST34", "room", "Infrastructure/room capacity component 34"),
}


REGISTRY_FILE = "cnes_capacity_registry.yaml"


def get_capacity_component(raw_field: str) -> CNESCapacityComponent:
    key = raw_field.upper()
    if key not in CAPACITY_COMPONENTS:
        raise CNESCapacityRegistryError(f"Unknown CNES capacity vector component: {raw_field}")
    return CAPACITY_COMPONENTS[key]


def require_vector_index(raw_field: str | None) -> CNESCapacityComponent:
    if raw_field is None or raw_field.strip().lower() in {"beds", "bed", "rooms", "room", "capacity", "generic_beds"}:
        raise CNESCapacityRegistryError("Generic CNES beds/capacity request is illegal without a capacity-vector index.")
    return get_capacity_component(raw_field)


def registry_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "registry": "cnes_capacity_vector",
        "components": {key: value.__dict__ for key, value in CAPACITY_COMPONENTS.items()},
    }


def capacity_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    return active_entries(REGISTRY_FILE, registry_root=registry_root)


def capacity_entry_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    return match_entry(field, capacity_entries(registry_root=registry_root))


def capacity_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    entry = capacity_entry_for_field(field, registry_root=registry_root)
    if entry is None:
        return None
    return {
        "registry": REGISTRY_FILE,
        "entry_id": entry.get("id"),
        "capacity_family": entry.get("capacity_family"),
        "capacity_index": entry.get("capacity_index"),
        "protected_non_equivalence": list(entry.get("protected_non_equivalence", []) or []),
    }
