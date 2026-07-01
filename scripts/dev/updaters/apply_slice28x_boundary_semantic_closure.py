from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dedent(text).lstrip(), encoding="utf-8")
    print(f"wrote {path}")


OUTPUT_TABLE_IO = r'''
"""Canonical output-table I/O over the PegaSUS storage boundary.

This module is intentionally small.  It gives production output attachers a
single row-oriented interface while keeping the parquet implementation inside
``pegasus.storage``.  Legacy fixture bundle writers may keep their historical
helpers until they are deleted or moved to a fixture namespace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa

from pegasus.storage import append_replace as storage_append_replace
from pegasus.storage import read_table as storage_read_table
from pegasus.storage import row_count as storage_row_count
from pegasus.storage import schema as storage_schema
from pegasus.storage import write_table as storage_write_table


def _rows(value: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(row) for row in (value or [])]


def _table_to_rows(table: pa.Table) -> list[dict[str, Any]]:
    return table.to_pylist()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a parquet table into row dictionaries through storage.read_table."""

    return _table_to_rows(storage_read_table(Path(path)))


def table_schema(path: str | Path) -> pa.Schema:
    """Return the persisted Arrow schema for an output artifact."""

    return storage_schema(Path(path))


def table_row_count(path: str | Path) -> int:
    """Return row count through the storage boundary."""

    return storage_row_count(Path(path))


def write_rows_like(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write rows, preserving an existing table schema when the file exists."""

    target = Path(path)
    materialized = _rows(rows)
    if target.exists():
        return storage_write_table(target, materialized, schema_policy="schema", schema=storage_schema(target))
    return storage_write_table(target, materialized, schema_policy="infer")


def empty_like(path: str | Path) -> Path:
    """Replace an existing table by an empty table with the same schema."""

    target = Path(path)
    return storage_write_table(target, [], schema_policy="schema", schema=storage_schema(target))


def append_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Append rows to an existing table while preserving its schema."""

    target = Path(path)
    incoming = _rows(rows)
    if not incoming:
        return target
    if not target.exists():
        return storage_write_table(target, incoming, schema_policy="infer")
    current = read_rows(target)
    return storage_write_table(target, current + incoming, schema_policy="schema", schema=storage_schema(target))


def _remove_existing(
    current: list[dict[str, Any]],
    *,
    id_column: str | None,
    incoming: list[dict[str, Any]],
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
    remove_column: str | None = None,
    remove_values: set[str] | None = None,
) -> list[dict[str, Any]]:
    incoming_ids = {str(row.get(id_column)) for row in incoming if id_column and row.get(id_column) is not None}
    explicit_ids = {str(value) for value in (remove_ids or set())}
    remove_values = {str(value) for value in (remove_values or set())}

    filtered: list[dict[str, Any]] = []
    for row in current:
        if id_column:
            value = str(row.get(id_column) or "")
            if value in incoming_ids or value in explicit_ids:
                continue
            if remove_prefixes and any(value.startswith(prefix) for prefix in remove_prefixes):
                continue
        if remove_column and remove_values and str(row.get(remove_column) or "") in remove_values:
            continue
        filtered.append(row)
    return filtered


def append_replace_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    id_column: str | None = None,
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
    remove_column: str | None = None,
    remove_values: set[str] | None = None,
) -> Path:
    """Append rows after removing existing rows selected by ID/prefix/value.

    This is a row-oriented adapter for run-bundle mutation.  It preserves an
    existing output schema when possible and delegates physical writes to
    ``pegasus.storage``.
    """

    target = Path(path)
    incoming = _rows(rows)
    if not target.exists():
        return storage_write_table(target, incoming, schema_policy="infer")
    current = read_rows(target)
    merged = _remove_existing(
        current,
        id_column=id_column,
        incoming=incoming,
        remove_ids=remove_ids,
        remove_prefixes=remove_prefixes,
        remove_column=remove_column,
        remove_values=remove_values,
    ) + incoming
    return storage_write_table(target, merged, schema_policy="schema", schema=storage_schema(target))


__all__ = [
    "append_replace_rows",
    "append_rows",
    "empty_like",
    "read_rows",
    "table_row_count",
    "table_schema",
    "write_rows_like",
]
'''

CORE_SEED = r'''
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
'''

BRIDGES = r'''
"""Metadata-only bridge planning for autonomous EFG expansion.

Bridge planning identifies cross-field semantic opportunities and blockers.  It
never materializes tensors; numeric realization remains the responsibility of
explicit compiler stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pegasus.core.hashing import content_hash
from pegasus.efg.core_seed import CoreSeed, build_core_seed_set


@dataclass(frozen=True)
class BridgeCandidate:
    bridge_id: str
    bridge_type: str
    numerator_id: str
    denominator_id: str | None
    required_operator: str
    support_relation: str
    registry_evidence: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "bridge_id": self.bridge_id,
            "bridge_type": self.bridge_type,
            "numerator_id": self.numerator_id,
            "denominator_id": self.denominator_id,
            "required_operator": self.required_operator,
            "support_relation": self.support_relation,
            "registry_evidence": list(self.registry_evidence),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BridgePlan:
    candidates: tuple[BridgeCandidate, ...]
    blocked: tuple[dict[str, Any], ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.candidates),
            "blocked_count": len(self.blocked),
            "bridge_type_counts": _counts(candidate.bridge_type for candidate in self.candidates),
            "candidates": [candidate.as_manifest() for candidate in self.candidates],
            "blocked": list(self.blocked),
        }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _bridge_id(kind: str, left: str, right: str | None) -> str:
    suffix = content_hash({"bridge_type": kind, "numerator": left, "denominator": right})[:16]
    return f"bridge::{kind}::{suffix}"


def _candidate(kind: str, numerator: CoreSeed, denominator: CoreSeed | None, *, operator: str = "ratio") -> BridgeCandidate:
    evidence = tuple(sorted(set(numerator.registry_evidence + ((denominator.registry_evidence if denominator else ())))))
    return BridgeCandidate(
        bridge_id=_bridge_id(kind, numerator.field_id, denominator.field_id if denominator else None),
        bridge_type=kind,
        numerator_id=numerator.field_id,
        denominator_id=denominator.field_id if denominator else None,
        required_operator=operator,
        support_relation="requires_alignment_or_transform",
        registry_evidence=evidence,
        warnings=("metadata_only_bridge_candidate",),
    )


def _by_role(seeds: Iterable[CoreSeed], role: str) -> list[CoreSeed]:
    return [seed for seed in seeds if seed.seed_role == role]


def plan_bridge_candidates(
    fields: Iterable[Any],
    *,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
) -> BridgePlan:
    seed_set = build_core_seed_set(fields, registry_root=registry_root, intent=intent)
    seeds = list(seed_set.seeds)
    candidates: list[BridgeCandidate] = []
    blocked: list[dict[str, Any]] = list(seed_set.blocked)

    populations = _by_role(seeds, "population_denominator_seed")
    deaths = _by_role(seeds, "death_event_seed")
    births = _by_role(seeds, "birth_event_seed")
    admissions = _by_role(seeds, "admission_event_seed")
    capacities = _by_role(seeds, "facility_capacity_seed")
    costs = _by_role(seeds, "cost_component_seed")

    for death in deaths:
        for population in populations:
            candidates.append(_candidate("mortality_rate_bridge", death, population))
    for birth in births:
        for population in populations:
            candidates.append(_candidate("birth_rate_bridge", birth, population))
    for admission in admissions:
        for population in populations:
            candidates.append(_candidate("admission_rate_bridge", admission, population))
        for capacity in capacities:
            candidates.append(_candidate("capacity_pressure_bridge", admission, capacity))
    for cost in costs:
        for admission in admissions:
            candidates.append(_candidate("cost_per_admission_bridge", cost, admission))

    if not populations and (deaths or births or admissions):
        blocked.append({"reason": "population_denominator_seed_missing", "affected_event_seed_count": len(deaths) + len(births) + len(admissions)})
    if admissions and not capacities:
        blocked.append({"reason": "facility_capacity_seed_missing_for_admission_bridge", "admission_seed_count": len(admissions)})

    return BridgePlan(candidates=tuple(candidates), blocked=tuple(blocked))


def bridge_summary(plan: BridgePlan) -> dict[str, Any]:
    return plan.as_manifest()


__all__ = ["BridgeCandidate", "BridgePlan", "bridge_summary", "plan_bridge_candidates"]
'''

GENERIC_REGISTRY = r'''
"""Generic YAML registry loader for thin registry wrapper modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from pegasus.core.exceptions import RegistryValidationError
from pegasus.core.hashing import sha256_file


@dataclass(frozen=True)
class RegistryEntry:
    id: str
    status: str
    description: str
    warnings: tuple[str, ...]
    payload: Mapping[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "description": self.description,
            "warnings": list(self.warnings),
            "payload": dict(self.payload),
        }


def _candidate_paths(registry_file: str | Sequence[str], root: str | Path) -> list[Path]:
    names = [registry_file] if isinstance(registry_file, str) else list(registry_file)
    return [Path(root) / name for name in names]


def resolve_registry_path(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> Path | None:
    candidates = _candidate_paths(registry_file, root)
    for path in candidates:
        if path.exists():
            return path
    if required:
        names = ", ".join(str(path) for path in candidates)
        raise RegistryValidationError(f"No registry file found among: {names}")
    return None


def load_registry_payload(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> dict[str, Any]:
    path = resolve_registry_path(registry_file, root=root, required=required)
    if path is None:
        return {"entries": [], "registry_file": None, "registry_missing": True}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RegistryValidationError(f"Registry payload must be a mapping: {path}")
    payload.setdefault("entries", [])
    payload["registry_file"] = str(path)
    payload["registry_sha256"] = sha256_file(path)
    return payload


def _entry(entry: Mapping[str, Any], index: int) -> RegistryEntry:
    entry_id = str(entry.get("id") or entry.get("name") or f"entry_{index}")
    warnings = entry.get("warnings") or ()
    if isinstance(warnings, str):
        warnings = (warnings,)
    return RegistryEntry(
        id=entry_id,
        status=str(entry.get("status") or "active"),
        description=str(entry.get("description") or entry_id),
        warnings=tuple(str(value) for value in warnings),
        payload=dict(entry),
    )


def load_entries(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> tuple[RegistryEntry, ...]:
    payload = load_registry_payload(registry_file, root=root, required=required)
    entries = payload.get("entries") or []
    if not isinstance(entries, list):
        raise RegistryValidationError("Registry entries must be a list")
    return tuple(_entry(entry, index) for index, entry in enumerate(entries) if isinstance(entry, Mapping))


def active_entries(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> tuple[RegistryEntry, ...]:
    return tuple(entry for entry in load_entries(registry_file, root=root, required=required) if entry.status == "active")


def get_entry(
    registry_file: str | Sequence[str],
    entry_id: str,
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> RegistryEntry:
    for entry in load_entries(registry_file, root=root, required=required):
        if entry.id == entry_id:
            return entry
    raise RegistryValidationError(f"Entry not found in registry {registry_file}: {entry_id}")


def registry_manifest(
    registry_file: str | Sequence[str],
    *,
    root: str | Path = "config/registries",
    required: bool = True,
) -> dict[str, Any]:
    payload = load_registry_payload(registry_file, root=root, required=required)
    return {
        "registry_file": payload.get("registry_file"),
        "schema_version": payload.get("schema_version"),
        "registry_version": payload.get("registry_version"),
        "entry_count": len(payload.get("entries") or []),
        "registry_sha256": payload.get("registry_sha256"),
        "registry_missing": bool(payload.get("registry_missing")),
    }


__all__ = [
    "RegistryEntry",
    "active_entries",
    "get_entry",
    "load_entries",
    "load_registry_payload",
    "registry_manifest",
    "resolve_registry_path",
]
'''

WRAPPER_TEMPLATE = r'''
"""Thin registry wrapper for {label} registry entries."""

from __future__ import annotations

from pathlib import Path

from pegasus.registries.generic import RegistryEntry, active_entries, get_entry, load_entries, registry_manifest

REGISTRY_FILES = {files!r}


def load_{stem}_entries(*, root: str | Path = "config/registries", required: bool = False) -> tuple[RegistryEntry, ...]:
    return load_entries(REGISTRY_FILES, root=root, required=required)


def active_{stem}_entries(*, root: str | Path = "config/registries", required: bool = False) -> tuple[RegistryEntry, ...]:
    return active_entries(REGISTRY_FILES, root=root, required=required)


def get_{stem}_entry(entry_id: str, *, root: str | Path = "config/registries", required: bool = True) -> RegistryEntry:
    return get_entry(REGISTRY_FILES, entry_id, root=root, required=required)


def {stem}_registry_manifest(*, root: str | Path = "config/registries", required: bool = False) -> dict:
    return registry_manifest(REGISTRY_FILES, root=root, required=required)


__all__ = [
    "REGISTRY_FILES",
    "load_{stem}_entries",
    "active_{stem}_entries",
    "get_{stem}_entry",
    "{stem}_registry_manifest",
]
'''

MANIFEST_WRAPPER = r'''
"""Thin accessors for the canonical registry manifest."""

from __future__ import annotations

from pathlib import Path

from pegasus.registries.generic import load_registry_payload, registry_manifest

REGISTRY_FILES = ("registry_manifest.yaml",)


def load_registry_manifest_payload(*, root: str | Path = "config/registries", required: bool = True) -> dict:
    return load_registry_payload(REGISTRY_FILES, root=root, required=required)


def canonical_registry_manifest(*, root: str | Path = "config/registries", required: bool = True) -> dict:
    return registry_manifest(REGISTRY_FILES, root=root, required=required)


__all__ = ["REGISTRY_FILES", "canonical_registry_manifest", "load_registry_manifest_payload"]
'''

AUDIT = r'''
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

STORAGE_PATTERNS = re.compile(r"pyarrow\.parquet|\bpq\.read_table\b|\bpq\.write_table\b|\bpl\.read_parquet\b|\bpl\.scan_parquet\b|\.write_parquet\(")
COMPUTE_PATTERNS = re.compile(r"torch\.cuda|torch\.device|manual_seed|np\.random\.seed|random\.seed")

STRICT_STORAGE_FILES = [
    "src/pegasus/efg/compile_attach.py",
    "src/pegasus/efg/promotion_apply.py",
    "src/pegasus/output/cnes_sih_compile_attach.py",
    "src/pegasus/output/maternal_child_compile_attach.py",
    "src/pegasus/output/population_tensor_compile_attach.py",
    "src/pegasus/output/race_bridge_attach.py",
    "src/pegasus/output/sidra_denominator_anchor.py",
    "src/pegasus/dashboard/read_only.py",
    "src/pegasus/acceptance/contracts.py",
]

STRICT_COMPUTE_FILES = [
    "src/pegasus/pirs/hsic.py",
    "src/pegasus/pirs/nulls.py",
    "src/pegasus/pirs/nystrom.py",
    "src/pegasus/pirs/rff.py",
    "src/pegasus/she/stdfm/torch_solver.py",
    "src/pegasus/she/population/torch_kernels.py",
    "src/pegasus/she/population/block_coordinate.py",
    "src/pegasus/she/population/sparse_admm.py",
]

REQUIRED_UNBLOCKED = [
    "src/pegasus/efg/core_seed.py",
    "src/pegasus/efg/bridges.py",
    "src/pegasus/registries/generic.py",
    "src/pegasus/registries/models.py",
    "src/pegasus/registries/residuals.py",
    "src/pegasus/registries/hsic.py",
    "src/pegasus/registries/nulls.py",
    "src/pegasus/registries/output.py",
    "src/pegasus/registries/events.py",
    "src/pegasus/registries/composite_decoders.py",
    "src/pegasus/registries/race_axis.py",
    "src/pegasus/registries/sidra.py",
    "src/pegasus/registries/manifest.py",
]


def _scan(paths: list[str], pattern: re.Pattern[str]) -> list[dict]:
    findings: list[dict] = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                findings.append({"path": rel, "line": number, "text": line.strip()})
    return findings


def run_audit() -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_UNBLOCKED:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"required file missing: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if "slice0_scaffold_only" in text or "BlockedModuleError" in text:
            errors.append(f"required module still scaffold-blocked: {rel}")

    storage_hits = _scan(STRICT_STORAGE_FILES, STORAGE_PATTERNS)
    compute_hits = _scan(STRICT_COMPUTE_FILES, COMPUTE_PATTERNS)

    # This first consolidation slice records production-boundary bypasses as
    # warnings rather than blocking every historical module at once.  The audit
    # is still useful because new/de-scaffolded files above are hard errors, and
    # future slices can promote these warnings to errors as files are migrated.
    for hit in storage_hits:
        warnings.append(f"storage bypass candidate: {hit['path']}:{hit['line']}: {hit['text']}")
    for hit in compute_hits:
        warnings.append(f"compute bypass candidate: {hit['path']}:{hit['line']}: {hit['text']}")

    return {
        "audit": "slice28x_production_boundaries",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "storage_bypass_count": len(storage_hits),
        "compute_bypass_count": len(compute_hits),
    }


def main() -> None:
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''

TEST_TABLE_IO = r'''
from __future__ import annotations

from pegasus.output.table_io import append_replace_rows, empty_like, read_rows, table_row_count, write_rows_like


def test_slice28x_table_io_write_read_append_replace_and_empty(tmp_path):
    path = tmp_path / "table.parquet"
    write_rows_like(path, [{"id": "a", "value": 1}, {"id": "b", "value": 2}])
    assert read_rows(path) == [{"id": "a", "value": 1}, {"id": "b", "value": 2}]

    append_replace_rows(path, [{"id": "b", "value": 20}, {"id": "c", "value": 3}], id_column="id")
    rows = read_rows(path)
    assert rows == [{"id": "a", "value": 1}, {"id": "b", "value": 20}, {"id": "c", "value": 3}]
    assert table_row_count(path) == 3

    empty_like(path)
    assert read_rows(path) == []
    assert table_row_count(path) == 0
'''

TEST_CORE_BRIDGES = r'''
from __future__ import annotations

from types import SimpleNamespace

from pegasus.efg.bridges import bridge_summary, plan_bridge_candidates
from pegasus.efg.core_seed import build_core_seed_set, classify_core_seed, core_seed_summary


def _lineage():
    return SimpleNamespace(registry_versions={"test_registry": "v1"}, source_manifest_hashes=["hash_a"])


def _field(field_id, *, carrier, unit, aggregation="additive", role=None, source=None, name=None):
    return SimpleNamespace(
        id=field_id,
        field_id=field_id,
        name=name or field_id,
        kind="extensive_measure",
        carrier=carrier,
        unit=unit,
        aggregation=aggregation,
        role=role or [],
        source=source or ["test"],
        lineage=_lineage(),
        warnings=[],
    )


def test_slice28x_core_seed_classifies_major_seed_roles():
    death = _field("sim_deaths", carrier="Deaths", unit="counts", source=["SIM"])
    population = _field("sidra_population", carrier="Population", unit="persons", source=["SIDRA"])
    diagnostic = _field("sim_CAUSABAS", carrier="Deaths", unit="ICD10", aggregation="non_aggregable", source=["SIM"])

    assert classify_core_seed(death).seed_role == "death_event_seed"
    assert classify_core_seed(population).seed_role == "population_denominator_seed"
    assert classify_core_seed(diagnostic).seed_role == "diagnostic_observer_seed"

    seed_set = build_core_seed_set([death, population, diagnostic])
    summary = core_seed_summary(seed_set)
    assert summary["seed_count"] == 3
    assert summary["role_counts"]["diagnostic_observer_seed"] == 1


def test_slice28x_bridge_planner_emits_metadata_bridge_candidates():
    fields = [
        _field("sim_deaths", carrier="Deaths", unit="counts", source=["SIM"]),
        _field("sidra_population", carrier="Population", unit="persons", source=["SIDRA"]),
        _field("sih_admissions", carrier="HospitalAdmissions", unit="counts", source=["SIH"]),
        _field("cnes_beds", carrier="Facilities", unit="beds", source=["CNES"]),
    ]
    plan = plan_bridge_candidates(fields)
    summary = bridge_summary(plan)
    assert summary["candidate_count"] >= 3
    assert summary["bridge_type_counts"]["mortality_rate_bridge"] == 1
    assert summary["bridge_type_counts"]["capacity_pressure_bridge"] == 1
'''

TEST_REGISTRY = r'''
from __future__ import annotations

from pathlib import Path

import yaml

from pegasus.registries.generic import active_entries, get_entry, registry_manifest
from pegasus.registries.models import active_model_entries
from pegasus.registries.nulls import active_null_entries


def _write_registry(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        yaml.safe_dump(
            {
                "schema_version": "1",
                "registry_version": "test",
                "entries": [
                    {"id": "active_entry", "status": "active", "description": "active", "warnings": []},
                    {"id": "deprecated_entry", "status": "deprecated", "description": "deprecated", "warnings": ["deprecated"]},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_slice28x_generic_registry_loader_reads_active_entries(tmp_path):
    _write_registry(tmp_path, "inference/model_registry.yaml")
    entries = active_entries("inference/model_registry.yaml", root=tmp_path)
    assert [entry.id for entry in entries] == ["active_entry"]
    assert get_entry("inference/model_registry.yaml", "deprecated_entry", root=tmp_path).status == "deprecated"
    assert registry_manifest("inference/model_registry.yaml", root=tmp_path)["entry_count"] == 2


def test_slice28x_registry_wrappers_import_and_delegate(tmp_path):
    _write_registry(tmp_path, "inference/model_registry.yaml")
    _write_registry(tmp_path, "ontology/null_registry.yaml")
    assert [entry.id for entry in active_model_entries(root=tmp_path)] == ["active_entry"]
    assert [entry.id for entry in active_null_entries(root=tmp_path)] == ["active_entry"]
'''

TEST_AUDIT = r'''
from __future__ import annotations

from scripts.dev.audits.audit_slice28x_production_boundaries import run_audit


def test_slice28x_production_boundary_audit_runs():
    result = run_audit()
    assert result["status"] == "passed"
    assert result["audit"] == "slice28x_production_boundaries"
'''

DOC = r'''
# Production Boundaries

PegaSUS production code must use explicit compiler boundaries rather than
private one-off table or device helpers.

Storage writes in production output, EFG, PIRS, acceptance, and dashboard code
should route through `pegasus.output.table_io`, which delegates physical parquet
operations to `pegasus.storage`.

Numerical device, seed, dtype, and memory policy should route through
`pegasus.compute`.

The EFG core seed and bridge modules are metadata-first. They classify seeds and
bridge opportunities but do not materialize tensors or silently execute numeric
fallbacks.

Legacy fixture bundle writers remain quarantined compatibility surfaces until
parity and deletion are explicitly executed.
'''


def wrapper_text(stem: str, label: str, files: tuple[str, ...]) -> str:
    return WRAPPER_TEMPLATE.format(stem=stem, label=label, files=files)


def main() -> None:
    write("src/pegasus/output/table_io.py", OUTPUT_TABLE_IO)
    write("src/pegasus/efg/core_seed.py", CORE_SEED)
    write("src/pegasus/efg/bridges.py", BRIDGES)
    write("src/pegasus/registries/generic.py", GENERIC_REGISTRY)

    wrappers = {
        "models": ("model", "model", ("inference/model_registry.yaml", "models.yaml")),
        "residuals": ("residual", "residual", ("inference/residual_registry.yaml", "residuals.yaml")),
        "hsic": ("hsic", "HSIC", ("inference/hsic_registry.yaml", "hsic.yaml")),
        "nulls": ("null", "null-regime", ("ontology/null_registry.yaml", "nulls_registry.yaml", "nulls.yaml")),
        "output": ("output", "output", ("output_registry.yaml", "output.yaml")),
        "events": ("event", "clinical event", ("health/clinical_event_definitions.yaml", "events_registry.yaml", "events.yaml")),
        "composite_decoders": ("composite_decoder", "composite decoder", ("composite_decoder_registry.yaml", "datasus/composite_decoders.yaml")),
        "race_axis": ("race_axis", "race axis", ("demographic/race_axis_registry.yaml", "race_axis.yaml")),
        "sidra": ("sidra", "SIDRA", ("sidra_registry.yaml", "sidra.yaml", "sidra_tables.yaml")),
    }
    for module, (stem, label, files) in wrappers.items():
        write(f"src/pegasus/registries/{module}.py", wrapper_text(stem, label, files))
    write("src/pegasus/registries/manifest.py", MANIFEST_WRAPPER)

    write("scripts/dev/audits/audit_slice28x_production_boundaries.py", AUDIT)
    write("tests/unit/test_slice28x_output_table_io.py", TEST_TABLE_IO)
    write("tests/unit/test_slice28x_core_seed_and_bridges.py", TEST_CORE_BRIDGES)
    write("tests/unit/test_slice28x_generic_registry_wrappers.py", TEST_REGISTRY)
    write("tests/integration/test_slice28x_production_boundary_audit.py", TEST_AUDIT)
    write("docs/production_boundaries.md", DOC)

    # Persist this updater in the repo-local updater directory for traceability.
    updater_target = ROOT / "scripts/dev/updaters/apply_slice28x_boundary_semantic_closure.py"
    updater_target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_text(encoding="utf-8")
    updater_target.write_text(source, encoding="utf-8")
    print(f"wrote {updater_target.relative_to(ROOT)}")

    print("Slice 28X updater completed. Run the targeted validation block next.")


if __name__ == "__main__":
    main()
