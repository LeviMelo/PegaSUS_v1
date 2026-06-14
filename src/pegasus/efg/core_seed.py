"""Core EFG seed classification for autonomous graph construction.

The seed layer is metadata-only.  It does not materialize tensors and does not
replace ``efg.dag``; it summarizes which substrate/admitted fields form the
compiler's V_core surface and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value if item is not None)
    except TypeError:
        return (str(value),)


def _field_attr(field: Any, name: str, default: Any = None) -> Any:
    if isinstance(field, dict):
        return field.get(name, default)
    return getattr(field, name, default)


def _lineage_attr(field: Any, name: str) -> Any:
    lineage = _field_attr(field, "lineage", None)
    if lineage is None:
        return None
    if isinstance(lineage, dict):
        return lineage.get(name)
    return getattr(lineage, name, None)


def _text_blob(field: Any) -> str:
    parts = [
        _field_attr(field, "id", ""),
        _field_attr(field, "field_id", ""),
        _field_attr(field, "name", ""),
        _field_attr(field, "kind", ""),
        _field_attr(field, "carrier", ""),
        _field_attr(field, "unit", ""),
        _field_attr(field, "aggregation", ""),
        " ".join(_as_tuple(_field_attr(field, "role", ()))),
        " ".join(_as_tuple(_field_attr(field, "source", ()))),
    ]
    return " ".join(str(part).lower() for part in parts if part is not None)


def _source_system(field: Any) -> str:
    source = _as_tuple(_field_attr(field, "source", ()))
    if source:
        return source[0]
    blob = _text_blob(field)
    for candidate in ("SIM", "SINASC", "SIH", "CNES", "SIDRA", "IBGE"):
        if candidate.lower() in blob:
            return candidate
    return "UNKNOWN"


def _seed_role(field: Any) -> str:
    blob = _text_blob(field)
    carrier = str(_field_attr(field, "carrier", "")).lower()
    unit = str(_field_attr(field, "unit", "")).lower()
    aggregation = str(_field_attr(field, "aggregation", "")).lower()

    if "icd" in unit or "diagn" in blob or aggregation == "non_aggregable":
        return "diagnostic_observer_seed"
    if "population" in blob or carrier in {"population", "persons"} or unit in {"person", "persons", "people"}:
        return "population_denominator_seed"
    if "birth" in blob or "livebirth" in blob or "live_birth" in blob:
        return "birth_event_seed"
    if "admission" in blob or carrier in {"hospitaladmissions", "admissions"}:
        return "admission_event_seed"
    if "bed" in blob or "capacity" in blob or carrier in {"facilities", "facility"}:
        return "facility_capacity_seed"
    if unit in {"brl", "currency"} or "cost" in blob or "val_" in blob:
        return "cost_component_seed"
    if "context" in blob or "sidra" in blob or "gradient" in blob:
        return "context_gradient_seed"
    if "death" in blob or carrier == "deaths":
        return "death_event_seed"
    if "bridge" in blob:
        return "bridge_candidate_seed"
    return "unknown_seed"


def _registry_evidence(field: Any) -> tuple[str, ...]:
    versions = _lineage_attr(field, "registry_versions") or {}
    if isinstance(versions, dict) and versions:
        return tuple(f"{key}={value}" for key, value in sorted(versions.items()))
    evidence = []
    for key in ("registry_id", "registry_entry", "topology_role", "capacity_family", "cost_component"):
        value = _field_attr(field, key, None)
        if value:
            evidence.append(f"{key}={value}")
    return tuple(evidence)


def _source_hashes(field: Any) -> tuple[str, ...]:
    hashes = _lineage_attr(field, "source_manifest_hashes")
    return _as_tuple(hashes)


@dataclass(frozen=True)
class CoreSeed:
    seed_id: str
    field_id: str
    source_system: str
    source_field: str | None
    seed_role: str
    carrier: str
    unit: str
    aggregation: str
    registry_evidence: tuple[str, ...]
    source_hashes: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "field_id": self.field_id,
            "source_system": self.source_system,
            "source_field": self.source_field,
            "seed_role": self.seed_role,
            "carrier": self.carrier,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "registry_evidence": list(self.registry_evidence),
            "source_hashes": list(self.source_hashes),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CoreSeedSet:
    seeds: tuple[CoreSeed, ...]
    blocked: tuple[dict[str, Any], ...]
    registry_root: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "registry_root": self.registry_root,
            "seed_count": len(self.seeds),
            "blocked_count": len(self.blocked),
            "role_counts": _role_counts(seed.seed_role for seed in self.seeds),
            "seeds": [seed.as_manifest() for seed in self.seeds],
            "blocked": list(self.blocked),
        }


def _role_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def classify_core_seed(field: Any, *, registry_root: str | Path = "config/registries") -> CoreSeed | None:
    field_id = str(_field_attr(field, "id", None) or _field_attr(field, "field_id", None) or _field_attr(field, "name", ""))
    if not field_id:
        return None
    role = _seed_role(field)
    warnings = tuple(_as_tuple(_field_attr(field, "warnings", ())))
    source_field = _field_attr(field, "source_field", None) or _field_attr(field, "source_column", None) or _field_attr(field, "name", None)
    return CoreSeed(
        seed_id=f"seed::{field_id}",
        field_id=field_id,
        source_system=_source_system(field),
        source_field=str(source_field) if source_field is not None else None,
        seed_role=role,
        carrier=str(_field_attr(field, "carrier", "unknown")),
        unit=str(_field_attr(field, "unit", "unknown")),
        aggregation=str(_field_attr(field, "aggregation", "unknown")),
        registry_evidence=_registry_evidence(field),
        source_hashes=_source_hashes(field),
        warnings=warnings,
    )


def build_core_seed_set(
    fields: Iterable[Any],
    *,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
) -> CoreSeedSet:
    del intent
    seeds: list[CoreSeed] = []
    blocked: list[dict[str, Any]] = []
    for index, field in enumerate(fields):
        seed = classify_core_seed(field, registry_root=registry_root)
        if seed is None:
            blocked.append({"index": index, "reason": "missing_field_identity"})
        else:
            seeds.append(seed)
    return CoreSeedSet(seeds=tuple(seeds), blocked=tuple(blocked), registry_root=str(registry_root))


def core_seed_summary(seed_set: CoreSeedSet) -> dict[str, Any]:
    return seed_set.as_manifest()


__all__ = ["CoreSeed", "CoreSeedSet", "build_core_seed_set", "classify_core_seed", "core_seed_summary"]
