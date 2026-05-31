from __future__ import annotations

from pegasus.core.exceptions import RegistryError
from pegasus.registries.loader import LoadedRegistrySet
from pegasus.registries.schemas import (
    AggregationRegistry,
    CarrierRegistry,
    QualityPermissionsRegistry,
    RaceAxisRegistry,
    SourceFieldsRegistry,
    UnitRegistry,
)


def _assert_unique_ids(name: str, ids: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for item in ids:
        if item in seen:
            duplicates.add(item)
        seen.add(item)

    if duplicates:
        raise RegistryError(f"Registry '{name}' has duplicate ids: {sorted(duplicates)}")


def validate_registry_set(registry_set: LoadedRegistrySet) -> list[str]:
    warnings: list[str] = []

    required = {
        "source_fields",
        "carrier_registry",
        "unit_registry",
        "aggregation_registry",
        "provenance_registry",
        "quality_permissions",
        "race_axis_registry",
    }

    missing = required - set(registry_set.registries)
    if missing:
        raise RegistryError(f"Missing required registries: {sorted(missing)}")

    for name, loaded in registry_set.registries.items():
        payload = loaded.payload

        if hasattr(payload, "entries"):
            ids = [entry.id for entry in payload.entries]  # type: ignore[attr-defined]
            _assert_unique_ids(name, ids)

    _validate_source_fields(registry_set)
    _validate_carrier_and_unit_roles(registry_set)
    _validate_quality_permissions(registry_set)
    _validate_race_axes(registry_set)

    return warnings


def _validate_source_fields(registry_set: LoadedRegistrySet) -> None:
    source = registry_set.registries["source_fields"].payload
    if not isinstance(source, SourceFieldsRegistry):
        raise RegistryError("source_fields registry has wrong type.")

    if not source.entries:
        raise RegistryError("source_fields registry must not be empty.")

    ids = {entry.id for entry in source.entries}

    required_first_slice = {
        "SIM-DO.DTOBITO",
        "SIM-DO.IDADE",
        "SIM-DO.SEXO",
        "SIM-DO.RACACOR",
        "SIM-DO.CODMUNRES",
        "SIM-DO.CAUSABAS",
    }

    missing = required_first_slice - ids
    if missing:
        raise RegistryError(
            "source_fields registry is missing first-slice SIM-DO fields: "
            f"{sorted(missing)}"
        )


def _validate_carrier_and_unit_roles(registry_set: LoadedRegistrySet) -> None:
    carrier = registry_set.registries["carrier_registry"].payload
    unit = registry_set.registries["unit_registry"].payload

    if not isinstance(carrier, CarrierRegistry):
        raise RegistryError("carrier_registry has wrong type.")
    if not isinstance(unit, UnitRegistry):
        raise RegistryError("unit_registry has wrong type.")

    carrier_roles = {entry.role for entry in carrier.entries if entry.legal}
    unit_roles = {role for entry in unit.entries for role in entry.legal_roles}

    missing_roles = carrier_roles - unit_roles
    if missing_roles:
        raise RegistryError(
            "Carrier roles missing compatible unit rules: "
            f"{sorted(missing_roles)}"
        )


def _validate_quality_permissions(registry_set: LoadedRegistrySet) -> None:
    quality = registry_set.registries["quality_permissions"].payload
    if not isinstance(quality, QualityPermissionsRegistry):
        raise RegistryError("quality_permissions has wrong type.")

    states = {entry.state for entry in quality.entries}
    required_states = {
        "verified",
        "fragile",
        "forced_fragile",
        "quarantined_descriptive",
        "quarantined_nochildren",
        "illegal_excluded",
        "blocked_solver_pending",
    }

    missing = required_states - states
    if missing:
        raise RegistryError(
            f"quality_permissions is missing required states: {sorted(missing)}"
        )


def _validate_race_axes(registry_set: LoadedRegistrySet) -> None:
    race_axis = registry_set.registries["race_axis_registry"].payload
    if not isinstance(race_axis, RaceAxisRegistry):
        raise RegistryError("race_axis_registry has wrong type.")

    by_source = {entry.source_system: entry for entry in race_axis.entries}

    for source in ("IBGE", "SIM-DO", "SIH-RD", "SINASC"):
        if source not in by_source:
            raise RegistryError(f"race_axis_registry missing source: {source}")

    if by_source["IBGE"].race_axis != "self_declared":
        raise RegistryError("IBGE race axis must be self_declared.")

    for source in ("SIM-DO", "SIH-RD", "SINASC"):
        entry = by_source[source]
        if entry.denominator_compatible:
            raise RegistryError(
                f"{source} race axis must not be denominator-compatible by default."
            )
        if entry.required_bridge_to_self_declared != "Bridge_R":
            raise RegistryError(
                f"{source} must require Bridge_R to align with self-declared denominators."
            )


def validate_aggregation_registry(registry_set: LoadedRegistrySet) -> None:
    agg = registry_set.registries["aggregation_registry"].payload
    if not isinstance(agg, AggregationRegistry):
        raise RegistryError("aggregation_registry has wrong type.")