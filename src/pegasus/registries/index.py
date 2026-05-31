from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegasus.core.exceptions import RegistryError
from pegasus.registries.loader import LoadedRegistrySet
from pegasus.registries.schemas import (
    CarrierEntry,
    CarrierRegistry,
    QualityPermissionEntry,
    QualityPermissionsRegistry,
    RaceAxisEntry,
    RaceAxisRegistry,
    UnitEntry,
    UnitRegistry,
)


@dataclass(frozen=True)
class RegistryIndex:
    """Convenience index over loaded registries.

    This does not replace the YAML registries. It creates fast lookup maps for
    legality checks and compiler operations.
    """

    carrier_by_key: dict[tuple[str, str, str], CarrierEntry]
    unit_by_key: dict[tuple[str, str, str], UnitEntry]
    race_axis_by_source: dict[str, RaceAxisEntry]
    quality_by_state: dict[str, QualityPermissionEntry]
    registry_hashes: dict[str, str]

    @classmethod
    def from_registry_set(cls, registry_set: LoadedRegistrySet) -> "RegistryIndex":
        carrier = registry_set.registries["carrier_registry"].payload
        unit = registry_set.registries["unit_registry"].payload
        race_axis = registry_set.registries["race_axis_registry"].payload
        quality = registry_set.registries["quality_permissions"].payload

        if not isinstance(carrier, CarrierRegistry):
            raise RegistryError("carrier_registry has wrong type.")
        if not isinstance(unit, UnitRegistry):
            raise RegistryError("unit_registry has wrong type.")
        if not isinstance(race_axis, RaceAxisRegistry):
            raise RegistryError("race_axis_registry has wrong type.")
        if not isinstance(quality, QualityPermissionsRegistry):
            raise RegistryError("quality_permissions has wrong type.")

        carrier_by_key: dict[tuple[str, str, str], CarrierEntry] = {}
        for entry in carrier.entries:
            key = (entry.numerator_carrier, entry.denominator_carrier, entry.role)
            carrier_by_key[key] = entry

        unit_by_key: dict[tuple[str, str, str], UnitEntry] = {}
        for entry in unit.entries:
            for role in entry.legal_roles:
                key = (entry.numerator_unit, entry.denominator_unit, role)
                unit_by_key[key] = entry

        race_axis_by_source = {entry.source_system: entry for entry in race_axis.entries}
        quality_by_state = {entry.state: entry for entry in quality.entries}

        return cls(
            carrier_by_key=carrier_by_key,
            unit_by_key=unit_by_key,
            race_axis_by_source=race_axis_by_source,
            quality_by_state=quality_by_state,
            registry_hashes=registry_set.hashes(),
        )

    def carrier_rule(
        self,
        numerator_carrier: str,
        denominator_carrier: str,
        role: str,
    ) -> CarrierEntry | None:
        return self.carrier_by_key.get((numerator_carrier, denominator_carrier, role))

    def unit_rule(
        self,
        numerator_unit: str,
        denominator_unit: str,
        role: str,
    ) -> UnitEntry | None:
        return self.unit_by_key.get((numerator_unit, denominator_unit, role))

    def race_axis_for_source(self, source_system: str) -> RaceAxisEntry | None:
        return self.race_axis_by_source.get(source_system)

    def permissions_for_state(self, state: str) -> dict[str, Any] | None:
        entry = self.quality_by_state.get(state)
        return None if entry is None else entry.permissions