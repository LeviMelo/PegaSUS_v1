from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.registries.source_fields import (
    SourceFieldRegistryEntry,
    normalize_source_system,
    resolve_source_field_entry,
    source_field_registry_summary,
)


@dataclass(frozen=True)
class SourceFieldSpec:
    """SHE-facing source-field semantics resolved from the registry layer."""

    source_system: str
    column_name: str
    carrier: str
    unit: str
    aggregation: str
    field_kind: str
    role: tuple[str, ...]
    quality_role: str
    provenance: tuple[str, ...]
    admissible: bool
    dashboard_safe: str
    axes: dict[str, Any]
    registry_hash: str
    warning: str | None = None
    matched_pattern: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "carrier": self.carrier,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "field_kind": self.field_kind,
            "role": list(self.role),
            "quality_role": self.quality_role,
            "provenance": list(self.provenance),
            "admissible": self.admissible,
            "dashboard_safe": self.dashboard_safe,
            "axes": self.axes,
            "registry_hash": self.registry_hash,
            "warning": self.warning,
            "matched_pattern": self.matched_pattern,
        }


@dataclass(frozen=True)
class SourceRegistryResolution:
    source_system: str
    column_name: str
    spec: SourceFieldSpec
    known: bool
    registry_backed: bool
    warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "known": self.known,
            "registry_backed": self.registry_backed,
            "warnings": list(self.warnings),
            "spec": self.spec.as_manifest(),
        }


def _spec_from_entry(entry: SourceFieldRegistryEntry) -> SourceFieldSpec:
    return SourceFieldSpec(
        source_system=entry.source_system,
        column_name=entry.column_name,
        carrier=entry.carrier,
        unit=entry.unit,
        aggregation=entry.aggregation,
        field_kind=entry.field_kind,
        role=entry.role,
        quality_role=entry.quality_role,
        provenance=entry.provenance,
        admissible=entry.admissible,
        dashboard_safe=entry.dashboard_safe,
        axes=dict(entry.axes),
        registry_hash=entry.registry_hash,
        warning=entry.warning,
        matched_pattern=entry.matched_pattern,
    )


def resolve_source_field(
    *,
    source_system: str,
    column_name: str,
    registry_root: str | Path = "config/registries",
) -> SourceRegistryResolution:
    system = normalize_source_system(source_system)
    entry = resolve_source_field_entry(source_system=system, column_name=column_name, registry_root=registry_root)
    spec = _spec_from_entry(entry)
    known = entry.warning != "unknown_source_field_requires_registry_entry"
    warnings: list[str] = []
    if entry.warning:
        warnings.append(entry.warning)
    if not entry.admissible:
        warnings.append(f"source_field_not_substrate_admissible:{entry.quality_role}")
    return SourceRegistryResolution(
        source_system=system,
        column_name=column_name,
        spec=spec,
        known=known,
        registry_backed=True,
        warnings=tuple(warnings),
    )


def source_registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    return source_field_registry_summary(registry_root=registry_root)
