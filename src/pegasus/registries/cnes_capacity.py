from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
        "components": {k: v.__dict__ for k, v in CAPACITY_COMPONENTS.items()},
    }
