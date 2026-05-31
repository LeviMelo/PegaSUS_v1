from __future__ import annotations

from dataclasses import dataclass

from pegasus.core.exceptions import RegistryError
from pegasus.registries.loader import LoadedRegistrySet
from pegasus.registries.schemas import SourceFieldEntry, SourceFieldsRegistry


@dataclass(frozen=True)
class SourceFieldIndex:
    by_id: dict[str, SourceFieldEntry]
    by_source_raw: dict[tuple[str, str], SourceFieldEntry]
    by_source_canonical: dict[tuple[str, str], SourceFieldEntry]

    @classmethod
    def from_registry_set(cls, registry_set: LoadedRegistrySet) -> "SourceFieldIndex":
        payload = registry_set.registries["source_fields"].payload

        if not isinstance(payload, SourceFieldsRegistry):
            raise RegistryError("source_fields registry has wrong type.")

        by_id: dict[str, SourceFieldEntry] = {}
        by_source_raw: dict[tuple[str, str], SourceFieldEntry] = {}
        by_source_canonical: dict[tuple[str, str], SourceFieldEntry] = {}

        for entry in payload.entries:
            by_id[entry.id] = entry
            by_source_raw[(entry.source_system, entry.raw_field.upper())] = entry
            by_source_canonical[(entry.source_system, entry.canonical_field)] = entry

        return cls(
            by_id=by_id,
            by_source_raw=by_source_raw,
            by_source_canonical=by_source_canonical,
        )

    def get_raw(self, source_system: str, raw_field: str) -> SourceFieldEntry | None:
        return self.by_source_raw.get((source_system, raw_field.upper()))

    def get_canonical(self, source_system: str, canonical_field: str) -> SourceFieldEntry | None:
        return self.by_source_canonical.get((source_system, canonical_field))

    def require_canonical(self, source_system: str, canonical_field: str) -> SourceFieldEntry:
        entry = self.get_canonical(source_system, canonical_field)
        if entry is None:
            raise RegistryError(
                f"Missing source field mapping for {source_system}.{canonical_field}"
            )
        return entry