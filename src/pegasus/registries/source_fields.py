from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash
from pegasus.registries.aggregation import get_aggregation
from pegasus.registries.carrier import get_carrier
from pegasus.registries.provenance import get_provenance
from pegasus.registries.quality import get_quality_role
from pegasus.registries.unit import get_unit


class SourceFieldRegistryError(ValueError):
    """Raised when source-field registry resolution fails."""


@dataclass(frozen=True)
class SourceFieldRegistryEntry:
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
    warning: str | None
    registry_hash: str
    matched_pattern: str | None = None
    decoder: str | None = None
    raw_fields: tuple[str, ...] = ()
    route: str | None = None
    parser: str | None = None
    output_key: str | None = None
    source_column_name: str | None = None

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
            "warning": self.warning,
            "registry_hash": self.registry_hash,
            "matched_pattern": self.matched_pattern,
            "decoder": self.decoder,
            "raw_fields": list(self.raw_fields),
            "route": self.route,
            "parser": self.parser,
            "output_key": self.output_key,
            "source_column_name": self.source_column_name,
        }


@dataclass(frozen=True)
class SourceFieldRegistry:
    registry_id: str
    schema_version: str
    registry_hash: str
    entries: dict[tuple[str, str], SourceFieldRegistryEntry]
    raw_routes: dict[tuple[str, str], tuple[SourceFieldRegistryEntry, ...]]
    patterns: tuple[tuple[str, re.Pattern[str], dict[str, Any]], ...]
    default_unknown: dict[str, Any]

    def resolve(self, *, source_system: str, column_name: str) -> SourceFieldRegistryEntry:
        system = normalize_source_system(source_system)
        key = (system, column_name)
        if key in self.entries:
            return self.entries[key]
        for pattern_system, pattern, spec in self.patterns:
            if pattern_system == system and pattern.fullmatch(column_name):
                return _entry_from_spec(
                    source_system=system,
                    column_name=column_name,
                    spec=spec,
                    registry_hash=self.registry_hash,
                    matched_pattern=pattern.pattern,
                )
        return _entry_from_spec(
            source_system=system,
            column_name=column_name,
            spec=self.default_unknown,
            registry_hash=self.registry_hash,
            matched_pattern=None,
        )

    def resolve_raw_all(self, *, source_system: str, raw_column_name: str) -> tuple[SourceFieldRegistryEntry, ...]:
        system = normalize_source_system(source_system)
        raw = str(raw_column_name)
        key = (system, raw.upper())
        if key in self.raw_routes:
            return self.raw_routes[key]
        return (
            _entry_from_spec(
                source_system=system,
                column_name=raw,
                spec=self.default_unknown,
                registry_hash=self.registry_hash,
                matched_pattern=None,
                source_column_name=raw,
            ),
        )

    def resolve_raw(self, *, source_system: str, raw_column_name: str) -> SourceFieldRegistryEntry:
        return self.resolve_raw_all(source_system=source_system, raw_column_name=raw_column_name)[0]


def normalize_source_system(value: str) -> str:
    token = str(value).strip().upper().replace("_", "-")
    aliases = {
        "SIM": "SIM-DO",
        "SIM-DO": "SIM-DO",
        "SIM-DO": "SIM-DO",
        "SINASC": "SINASC",
        "SIH": "SIH-RD",
        "SIH-RD": "SIH-RD",
        "CNES": "CNES-ST",
        "CNES-ST": "CNES-ST",
        "SIDRA": "SIDRA",
    }
    return aliases.get(token, token)


def _list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(x) for x in value)
    return (str(value),)


def _entry_from_spec(
    *,
    source_system: str,
    column_name: str,
    spec: dict[str, Any],
    registry_hash: str,
    matched_pattern: str | None,
    source_column_name: str | None = None,
) -> SourceFieldRegistryEntry:
    return SourceFieldRegistryEntry(
        source_system=source_system,
        column_name=column_name,
        carrier=str(spec.get("carrier", "AuditMetadata")),
        unit=str(spec.get("unit", "none")),
        aggregation=str(spec.get("aggregation", "non_aggregable")),
        field_kind=str(spec.get("field_kind", "observer_proxy")),
        role=_list(spec.get("role", [])),
        quality_role=str(spec.get("quality_role", "audit_only")),
        provenance=_list(spec.get("provenance", [])),
        admissible=bool(spec.get("admissible", False)),
        dashboard_safe=str(spec.get("dashboard_safe", "False")),
        axes=dict(spec.get("axes", {}) or {}),
        warning=None if spec.get("warning") is None else str(spec.get("warning")),
        registry_hash=registry_hash,
        matched_pattern=matched_pattern,
        decoder=None if spec.get("decoder") is None else str(spec.get("decoder")),
        raw_fields=_list(spec.get("raw_fields", [])),
        route=None if spec.get("route") is None else str(spec.get("route")),
        parser=None if spec.get("parser") is None else str(spec.get("parser")),
        output_key=None if spec.get("output_key") is None else str(spec.get("output_key")),
        source_column_name=source_column_name,
    )


def _validate_entry(entry: SourceFieldRegistryEntry, *, registry_root: str | Path) -> None:
    carrier = get_carrier(entry.carrier, registry_root=registry_root)
    unit = get_unit(entry.unit, registry_root=registry_root)
    get_aggregation(entry.aggregation, registry_root=registry_root)
    get_quality_role(entry.quality_role, registry_root=registry_root)
    for tag in entry.provenance:
        get_provenance(tag, registry_root=registry_root)
    if entry.unit not in carrier.allowed_units:
        raise SourceFieldRegistryError(
            f"source field {entry.source_system}.{entry.column_name} uses unit {entry.unit!r} not allowed for carrier {entry.carrier!r}"
        )
    if entry.dashboard_safe not in {"True", "False", "warning"}:
        raise SourceFieldRegistryError(
            f"source field {entry.source_system}.{entry.column_name} has invalid dashboard_safe={entry.dashboard_safe!r}"
        )


def load_source_field_registry(registry_root: str | Path = "config/registries") -> SourceFieldRegistry:
    # Cache the fully-built registry object keyed by (root, mtime). Previously this
    # re-parsed and rebuilt the entire registry on every resolve_source_field call
    # — i.e. once per column per record — making SHE normalization unusably slow.
    path = Path(registry_root) / "datasus/source_fields.yaml"
    mtime = path.stat().st_mtime if path.exists() else 0.0
    return _load_source_field_registry_cached(str(Path(registry_root)), mtime)


@lru_cache(maxsize=16)
def _load_source_field_registry_cached(registry_root: str, mtime: float) -> SourceFieldRegistry:
    path = Path(registry_root) / "datasus/source_fields.yaml"
    payload = load_yaml(path)
    registry_hash = content_hash(payload)
    source_systems = payload.get("source_systems", {})
    if not isinstance(source_systems, dict) or not source_systems:
        raise SourceFieldRegistryError(f"source field registry is empty or invalid: {path}")
    default_unknown = dict(payload.get("default_unknown", {}) or {})
    entries: dict[tuple[str, str], SourceFieldRegistryEntry] = {}
    raw_route_lists: dict[tuple[str, str], list[SourceFieldRegistryEntry]] = {}
    patterns: list[tuple[str, re.Pattern[str], dict[str, Any]]] = []
    for source_system, raw_system in source_systems.items():
        system = normalize_source_system(str(source_system))
        if not isinstance(raw_system, dict):
            raise SourceFieldRegistryError(f"source system registry block must be a mapping: {source_system}")
        for column_name, spec in (raw_system.get("fields", {}) or {}).items():
            if not isinstance(spec, dict):
                raise SourceFieldRegistryError(f"field spec must be a mapping: {source_system}.{column_name}")
            entry = _entry_from_spec(
                source_system=system,
                column_name=str(column_name),
                spec=spec,
                registry_hash=registry_hash,
                matched_pattern=None,
            )
            _validate_entry(entry, registry_root=registry_root)
            entries[(system, str(column_name))] = entry
            for raw_field in entry.raw_fields:
                raw_key = (system, str(raw_field).upper())
                raw_route_lists.setdefault(raw_key, []).append(_entry_from_spec(
                    source_system=system,
                    column_name=str(column_name),
                    spec=spec,
                    registry_hash=registry_hash,
                    matched_pattern=None,
                    source_column_name=str(raw_field),
                ))
        for pattern, spec in (raw_system.get("field_patterns", {}) or {}).items():
            if not isinstance(spec, dict):
                raise SourceFieldRegistryError(f"field pattern spec must be a mapping: {source_system}.{pattern}")
            compiled = re.compile(str(pattern))
            probe_name = "QTINST01" if "QTINST" in str(pattern) else ("QTLEIT01" if "QTLEIT" in str(pattern) else "example_state")
            probe = _entry_from_spec(source_system=system, column_name=probe_name, spec=spec, registry_hash=registry_hash, matched_pattern=str(pattern))
            _validate_entry(probe, registry_root=registry_root)
            patterns.append((system, compiled, spec))
    unknown_probe = _entry_from_spec(source_system="UNKNOWN", column_name="unknown", spec=default_unknown, registry_hash=registry_hash, matched_pattern=None)
    _validate_entry(unknown_probe, registry_root=registry_root)
    return SourceFieldRegistry(
        registry_id=str(payload.get("registry_id", "source_field_registry")),
        schema_version=str(payload.get("schema_version", "1.0")),
        registry_hash=registry_hash,
        entries=entries,
        raw_routes={key: tuple(value) for key, value in raw_route_lists.items()},
        patterns=tuple(patterns),
        default_unknown=default_unknown,
    )


@lru_cache(maxsize=8192)
def _resolve_source_field_entry_cached(source_system: str, column_name: str, registry_root: str) -> SourceFieldRegistryEntry:
    return load_source_field_registry(registry_root).resolve(source_system=source_system, column_name=column_name)


def resolve_source_field_entry(*, source_system: str, column_name: str, registry_root: str | Path = "config/registries") -> SourceFieldRegistryEntry:
    # Per (system, column) resolution is deterministic for a static registry and
    # is called once per column per record during normalization; memoize it so a
    # whole-state normalize does O(distinct_columns) registry lookups, not O(cells).
    return _resolve_source_field_entry_cached(source_system, column_name, str(registry_root))


@lru_cache(maxsize=8192)
def _resolve_raw_source_field_entry_cached(source_system: str, raw_column_name: str, registry_root: str) -> SourceFieldRegistryEntry:
    return load_source_field_registry(registry_root).resolve_raw(source_system=source_system, raw_column_name=raw_column_name)


def resolve_raw_source_field_entry(
    *,
    source_system: str,
    raw_column_name: str,
    registry_root: str | Path = "config/registries",
) -> SourceFieldRegistryEntry:
    return _resolve_raw_source_field_entry_cached(source_system, raw_column_name, str(registry_root))


def resolve_raw_source_field_entries(
    *,
    source_system: str,
    raw_column_name: str,
    registry_root: str | Path = "config/registries",
) -> tuple[SourceFieldRegistryEntry, ...]:
    return _resolve_raw_source_field_entries_cached(source_system, raw_column_name, str(registry_root))


@lru_cache(maxsize=8192)
def _resolve_raw_source_field_entries_cached(source_system: str, raw_column_name: str, registry_root: str) -> tuple[SourceFieldRegistryEntry, ...]:
    return load_source_field_registry(registry_root).resolve_raw_all(source_system=source_system, raw_column_name=raw_column_name)


def source_field_registry_summary(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_source_field_registry(registry_root)
    by_system: dict[str, int] = {}
    admissible = 0
    audit_only = 0
    for entry in registry.entries.values():
        by_system[entry.source_system] = by_system.get(entry.source_system, 0) + 1
        if entry.admissible:
            admissible += 1
        else:
            audit_only += 1
    return {
        "registry_id": registry.registry_id,
        "schema_version": registry.schema_version,
        "registry_hash": registry.registry_hash,
        "entry_count": len(registry.entries),
        "raw_route_count": sum(len(entries) for entries in registry.raw_routes.values()),
        "pattern_count": len(registry.patterns),
        "admissible_entry_count": admissible,
        "audit_or_excluded_entry_count": audit_only,
        "by_source_system": dict(sorted(by_system.items())),
    }


def registry_manifest(registry_root: str | Path = "config/registries") -> dict[str, Any]:
    registry = load_source_field_registry(registry_root)
    return {
        "registry_id": registry.registry_id,
        "schema_version": registry.schema_version,
        "registry_hash": registry.registry_hash,
        "entries": [entry.as_manifest() for entry in sorted(registry.entries.values(), key=lambda e: (e.source_system, e.column_name))],
        "raw_routes": [
            entry.as_manifest()
            for routes in registry.raw_routes.values()
            for entry in sorted(routes, key=lambda e: (e.source_system, e.source_column_name or "", e.column_name))
        ],
        "patterns": [{"source_system": s, "pattern": p.pattern} for s, p, _ in registry.patterns],
        "default_unknown": registry.default_unknown,
    }
