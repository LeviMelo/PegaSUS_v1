"""SHE-facing source-field registry resolution.

This module is the compatibility boundary between the YAML-backed registry layer
introduced in Slice 13B and the Slice 13A SHE substrate API.  It exposes canonical
registry carrier IDs for new code while retaining equality compatibility with the
legacy lowercase carrier checks used by early SHE tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pegasus.registries.source_fields import (
    SourceFieldRegistryEntry,
    normalize_source_system,
    registry_manifest as source_field_registry_manifest,
    resolve_source_field_entry,
    source_field_registry_summary,
)


DIAGNOSTIC_CODE_COLUMNS: frozenset[str] = frozenset({
    "underlying_icd_norm",
    "associated_conditions_norm",
    "principal_icd_norm",
    "anomaly_icd_code",
})


class CarrierId(str):
    """Canonical registry carrier ID with legacy equality compatibility.

    The visible value remains canonical, e.g. ``str(CarrierId("Deaths"))`` is
    ``"Deaths"``.  Equality also accepts the old Slice 13A lowercase/snake-case
    tokens, e.g. ``CarrierId("Deaths") == "deaths"`` and
    ``CarrierId("HospitalAdmissions") == "hospital_admissions"``.
    """

    def __new__(cls, value: object) -> "CarrierId":
        return str.__new__(cls, str(value))

    @staticmethod
    def _norm(value: object) -> str:
        text = str(value).replace("_", "").replace("-", "")
        return text.lower()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._norm(self) == self._norm(other)
        return str.__eq__(self, other)

    def __hash__(self) -> int:
        return hash(self._norm(self))


@dataclass(frozen=True)
class SourceFieldSpec:
    """SHE-facing source-field semantics resolved from YAML-backed registries."""

    source_system: str
    column_name: str
    carrier: CarrierId
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
    registry_carrier: str | None = None

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
        return self.admissible

    @property
    def registry_reason(self) -> str | None:
        if self.admissible:
            return None
        if self.warning:
            return self.warning
        if self.quality_role in {"audit_only", "sentinel_state"}:
            return "registry_not_admissible"
        return f"registry_not_admissible:{self.quality_role}"

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["carrier"] = str(self.carrier)
        payload["registry_carrier"] = self.registry_carrier or str(self.carrier)
        payload["column"] = self.column
        payload["technical_name"] = self.technical_name
        payload["substrate_kind"] = self.substrate_kind
        payload["admissible_by_registry"] = self.admissible_by_registry
        payload["registry_reason"] = self.registry_reason
        return payload


@dataclass(frozen=True)
class SourceRegistryResolution:
    source_system: str
    column_name: str
    spec: SourceFieldSpec
    known: bool
    registry_backed: bool
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "column_name": self.column_name,
            "spec": self.spec.as_manifest(),
            "known": self.known,
            "registry_backed": self.registry_backed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SourceRegistryBatchResolution:
    source_system: str
    registry_hash: str
    specs: tuple[SourceFieldSpec, ...]
    unresolved_columns: tuple[str, ...] = ()
    registry_backed: bool = True

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "registry_hash": self.registry_hash,
            "registry_backed": self.registry_backed,
            "specs": [spec.as_manifest() for spec in self.specs],
            "unresolved_columns": list(self.unresolved_columns),
        }


def _tuple(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _is_unknown_entry(entry: SourceFieldRegistryEntry) -> bool:
    return (
        getattr(entry, "warning", None) == "unknown_source_field_requires_registry_entry"
        and getattr(entry, "carrier", None) == "AuditMetadata"
        and getattr(entry, "admissible", None) is False
    )


def _spec_from_entry(entry: SourceFieldRegistryEntry) -> SourceFieldSpec:
    column_name = str(entry.column_name)
    unit = str(entry.unit)
    aggregation = str(entry.aggregation)
    role = _tuple(entry.role)
    warning = entry.warning

    if column_name in DIAGNOSTIC_CODE_COLUMNS or str(entry.quality_role) == "diagnostic_code":
        unit = "ICD10"
        aggregation = "non_aggregable"
        if "diagnostic_topology" not in role:
            role = tuple(dict.fromkeys(role + ("diagnostic_topology",)))
        if warning and "diagnostic_code_unit_normalized_to_ICD10" not in warning:
            warning = f"{warning};diagnostic_code_unit_normalized_to_ICD10"
        elif not warning and (entry.unit != "ICD10" or entry.aggregation != "non_aggregable"):
            warning = "diagnostic_code_unit_normalized_to_ICD10"

    registry_carrier = str(entry.carrier)
    return SourceFieldSpec(
        source_system=normalize_source_system(entry.source_system),
        column_name=column_name,
        carrier=CarrierId(registry_carrier),
        registry_carrier=registry_carrier,
        unit=unit,
        aggregation=aggregation,
        field_kind=str(entry.field_kind),
        role=role,
        quality_role=str(entry.quality_role),
        provenance=_tuple(entry.provenance),
        admissible=bool(entry.admissible),
        dashboard_safe=str(entry.dashboard_safe),
        axes=dict(entry.axes),
        registry_hash=str(entry.registry_hash),
        warning=warning,
        matched_pattern=entry.matched_pattern,
    )


def resolve_source_field(
    *,
    source_system: str,
    column_name: str,
    registry_root: str | Path = "config/registries",
) -> SourceRegistryResolution:
    """Resolve a single normalized source column against the source-field registry."""

    normalized = normalize_source_system(source_system)
    entry = resolve_source_field_entry(
        source_system=normalized,
        column_name=column_name,
        registry_root=registry_root,
    )
    spec = _spec_from_entry(entry)
    known = not _is_unknown_entry(entry)
    warnings: list[str] = []
    if spec.warning:
        warnings.append(spec.warning)
    if not spec.admissible:
        warnings.append(f"source_field_not_substrate_admissible:{spec.quality_role}")
    return SourceRegistryResolution(
        source_system=normalized,
        column_name=column_name,
        spec=spec,
        known=known,
        registry_backed=True,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def source_registry_hash(
    source_system: str | None = None,
    registry_root: str | Path = "config/registries",
) -> str:
    manifest = source_field_registry_summary(registry_root=registry_root)
    root_hash = str(manifest.get("registry_hash", "source_field_registry_v1"))
    if source_system is None:
        return root_hash
    normalized = normalize_source_system(source_system)
    return f"{root_hash}:{normalized}"


def resolve_source_fields(
    *,
    source_system: str,
    columns: list[str] | tuple[str, ...],
    allow_heuristic: bool = True,
    registry_root: str | Path = "config/registries",
) -> SourceRegistryBatchResolution:
    """Resolve multiple source columns.

    ``allow_heuristic`` is retained for the Slice 13A substrate API.  Slice 13B
    no longer emits heuristic analytical fields for unknown columns; unknowns are
    represented by the registry's explicit audit-only default entry.
    """

    del allow_heuristic
    normalized = normalize_source_system(source_system)
    specs: list[SourceFieldSpec] = []
    unresolved: list[str] = []
    for column in columns:
        result = resolve_source_field(
            source_system=normalized,
            column_name=str(column),
            registry_root=registry_root,
        )
        specs.append(result.spec)
        if not result.known:
            unresolved.append(str(column))
    return SourceRegistryBatchResolution(
        source_system=normalized,
        registry_hash=source_registry_hash(normalized, registry_root=registry_root),
        specs=tuple(specs),
        unresolved_columns=tuple(unresolved),
        registry_backed=True,
    )


def source_system_from_path(path: str | Path) -> str | None:
    text = str(path).replace("\\", "/").lower()
    checks: tuple[tuple[str, str], ...] = (
        ("sim", "SIM-DO"),
        ("sinasc", "SINASC"),
        ("sih", "SIH-RD"),
        ("cnes", "CNES-ST"),
        ("sidra", "SIDRA"),
    )
    for needle, source_system in checks:
        if needle in text:
            return source_system
    return None


def source_registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    """Return SHE-facing source-field registry metadata.

    Slice 13B tests and CLI callers consume summary counters at the top level.
    The full registry manifest is still preserved, and the same summary is also
    retained under ``summary`` for structured consumers.
    """
    summary = source_field_registry_summary(registry_root=registry_root)
    payload = dict(source_field_registry_manifest(registry_root=registry_root))
    payload.update(summary)
    payload["summary"] = dict(summary)
    payload["she_source_registry_api"] = {
        "registry_backed": True,
        "carrier_surface": "canonical_registry_id_with_legacy_equality",
        "batch_signature": "resolve_source_fields(source_system, columns, allow_heuristic=True, registry_root=...)",
    }
    return payload
