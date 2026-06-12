
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.registries.source_fields import (
    SourceFieldRegistryEntry,
    normalize_source_system,
    resolve_source_field_entry,
    source_field_registry_summary,
)


@dataclass(frozen=True)
class SourceFieldSpec:
    """SHE-facing source-field semantics resolved from YAML-backed registries.

    Slice 13B keeps the public Slice 13A compatibility attributes used by
    pegasus.she.substrate while exposing the richer registry fields introduced
    by the YAML source-field registry.
    """

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

    @property
    def column(self) -> str:
        return self.column_name

    @property
    def technical_name(self) -> str:
        return f"{self.source_system}.{self.column_name}"

    @property
    def substrate_kind(self) -> str:
        return self.field_kind

    @property
    def admissible_by_registry(self) -> bool:
        return bool(self.admissible)

    @property
    def registry_reason(self) -> str | None:
        if self.admissible:
            return None
        if self.quality_role in {"audit_only", "sentinel_state"}:
            return "registry_not_admissible"
        if self.warning:
            return self.warning
        return f"registry_not_admissible:{self.quality_role}"

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "column": self.column,
            "technical_name": self.technical_name,
            "carrier": self.carrier,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "field_kind": self.field_kind,
            "substrate_kind": self.substrate_kind,
            "role": list(self.role),
            "quality_role": self.quality_role,
            "provenance": list(self.provenance),
            "admissible": self.admissible,
            "admissible_by_registry": self.admissible_by_registry,
            "dashboard_safe": self.dashboard_safe,
            "axes": self.axes,
            "registry_hash": self.registry_hash,
            "registry_reason": self.registry_reason,
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


@dataclass(frozen=True)
class SourceRegistryBatchResolution:
    source_system: str
    registry_hash: str
    specs: tuple[SourceFieldSpec, ...]
    unresolved_columns: tuple[str, ...]
    registry_backed: bool = True

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "registry_hash": self.registry_hash,
            "registry_backed": self.registry_backed,
            "specs": [s.as_manifest() for s in self.specs],
            "unresolved_columns": list(self.unresolved_columns),
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


def source_registry_hash(source_system: str | None = None, registry_root: str | Path = "config/registries") -> str:
    summary = source_field_registry_summary(registry_root=registry_root)
    if source_system is None:
        return str(summary.get("registry_hash") or content_hash(summary))
    system = normalize_source_system(source_system)
    return content_hash({
        "registry_hash": summary.get("registry_hash"),
        "source_system": system,
        "entry_count": summary.get("by_source_system", {}).get(system, 0),
    })


def resolve_source_fields(
    *,
    source_system: str,
    columns: list[str],
    allow_heuristic: bool = True,
    registry_root: str | Path = "config/registries",
) -> SourceRegistryBatchResolution:
    system = normalize_source_system(source_system)
    specs: list[SourceFieldSpec] = []
    unresolved: list[str] = []
    for column in columns:
        resolution = resolve_source_field(source_system=system, column_name=column, registry_root=registry_root)
        if not resolution.known and not allow_heuristic:
            unresolved.append(column)
            continue
        specs.append(resolution.spec)
    return SourceRegistryBatchResolution(
        source_system=system,
        registry_hash=source_registry_hash(system, registry_root=registry_root),
        specs=tuple(specs),
        unresolved_columns=tuple(unresolved),
        registry_backed=True,
    )


def source_system_from_path(path: str | Path) -> str | None:
    text = str(path).replace("\\", "/").upper()
    for token, system in (
        ("SIM-DO", "SIM-DO"),
        ("SIM_DO", "SIM-DO"),
        ("/SIM/", "SIM-DO"),
        ("SINASC", "SINASC"),
        ("SIH-RD", "SIH-RD"),
        ("SIH_RD", "SIH-RD"),
        ("/SIH/", "SIH-RD"),
        ("CNES-ST", "CNES-ST"),
        ("CNES_ST", "CNES-ST"),
        ("CNES", "CNES-ST"),
        ("SIDRA", "SIDRA"),
    ):
        if token in text:
            return system
    return None


def source_registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    return source_field_registry_summary(registry_root=registry_root)
