from __future__ import annotations

import hashlib
import shutil
import textwrap
from pathlib import Path

import yaml

SLICE = "27A/27B repair v1"


def find_root() -> Path:
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "pegasus").is_dir() and (candidate / "config" / "registries").is_dir():
            return candidate
    raise SystemExit("Could not locate PegaSUS repository root. Run this script from inside C:\\Users\\Galaxy\\LEVI\\PegaSUS.")


ROOT = find_root()


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


SEMANTIC_MODULE = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.registries.loader import load_registry_file


def registry_root_path(root: str | Path = "config/registries") -> Path:
    return Path(root)


def registry_entries(name: str, *, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    payload = load_registry_file(Path(registry_root) / name)
    entries = payload.get("entries", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def active_entries(name: str, *, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    entries = registry_entries(name, registry_root=registry_root)
    return [
        entry
        for entry in entries
        if str(entry.get("status", "")).startswith("active")
        or str(entry.get("status", "")) in {"stable", "planned_contract", "experimental"}
    ]


def field_text(field: Any) -> str:
    parts: list[str] = []
    for attr in (
        "field_id",
        "id",
        "name",
        "source",
        "source_json",
        "role",
        "role_json",
        "metadata_json",
        "operator",
        "carrier",
        "unit",
        "aggregation",
    ):
        value = getattr(field, attr, None)
        if value is not None:
            parts.append(str(value))
    try:
        dump = field.model_dump()
    except Exception:
        dump = None
    if isinstance(dump, dict):
        parts.extend(str(value) for value in dump.values() if value is not None)
    return " ".join(parts).upper()


def match_entry(field: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most specific registry entry matching a field descriptor.

    Slice 27A originally returned the first matching entry.  That was too broad:
    a generic CNES pattern matched before QTLEIT/QTINST capacity-vector patterns,
    causing QTLEIT fields to be classified as facility_stock.  Specificity is now
    defined by the longest matching pattern, with entry-id matches treated as high
    specificity.  This preserves deterministic behavior while avoiding generic
    registry entries shadowing specialized semantics.
    """
    haystack = field_text(field)
    best: tuple[int, int, dict[str, Any]] | None = None
    for order, entry in enumerate(entries):
        score = -1
        patterns = entry.get("field_patterns", []) or []
        for pattern in patterns:
            token = str(pattern).upper()
            if token and token in haystack:
                score = max(score, len(token))
        entry_id = str(entry.get("id", "")).upper()
        if entry_id and entry_id in haystack:
            score = max(score, len(entry_id) + 1000)
        if score >= 0 and (best is None or score > best[0] or (score == best[0] and order < best[1])):
            best = (score, order, entry)
    return None if best is None else best[2]


def registry_is_scaffold_only(name: str, *, registry_root: str | Path = "config/registries") -> bool:
    entries = registry_entries(name, registry_root=registry_root)
    if not entries:
        return True
    if len(entries) == 1:
        entry = entries[0]
        warnings = {str(value) for value in entry.get("warnings", []) or []}
        status = str(entry.get("status", ""))
        return status == "deferred" and "scaffold_only" in warnings
    return all("scaffold_only" in {str(value) for value in entry.get("warnings", []) or []} for entry in entries)
'''


CNES_CAPACITY_MODULE = r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pegasus.registries.semantic import active_entries, match_entry


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


REGISTRY_FILE = "health/cnes_capacity_registry.yaml"


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
        "components": {key: value.__dict__ for key, value in CAPACITY_COMPONENTS.items()},
    }


def capacity_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    return active_entries(REGISTRY_FILE, registry_root=registry_root)


def capacity_entry_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    return match_entry(field, capacity_entries(registry_root=registry_root))


def capacity_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    entry = capacity_entry_for_field(field, registry_root=registry_root)
    if entry is None:
        return None
    return {
        "registry": REGISTRY_FILE,
        "entry_id": entry.get("id"),
        "capacity_family": entry.get("capacity_family"),
        "capacity_index": entry.get("capacity_index"),
        "protected_non_equivalence": list(entry.get("protected_non_equivalence", []) or []),
    }
'''


SIH_COST_MODULE = r'''
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.registries.semantic import active_entries, match_entry


class SIHCostRegistryError(ValueError):
    """Raised when SIH economic estimands collapse distinct billing components."""


@dataclass(frozen=True)
class SIHCostComponent:
    raw_field: str
    component_id: str
    carrier: str
    unit: str
    label: str


COST_COMPONENTS: dict[str, SIHCostComponent] = {
    "VAL_SH": SIHCostComponent("VAL_SH", "hospital_service_cost", "HospitalCosts_SH", "reais_hospital_services", "Hospital service billing component"),
    "VAL_SP": SIHCostComponent("VAL_SP", "professional_service_cost", "HospitalCosts_SP", "reais_professional_services", "Professional service billing component"),
    "VAL_UTI": SIHCostComponent("VAL_UTI", "icu_cost", "HospitalCosts_UTI", "reais_icu_services", "ICU billing component"),
    "VAL_TOT": SIHCostComponent("VAL_TOT", "total_admission_cost", "HospitalCosts_TOT", "reais_total_billing", "Total admission billing burden"),
}


REGISTRY_FILE = "health/sih_cost_registry.yaml"


def get_cost_component(raw_field: str) -> SIHCostComponent:
    key = raw_field.upper()
    if key not in COST_COMPONENTS:
        raise SIHCostRegistryError(f"Unknown SIH cost component: {raw_field}")
    return COST_COMPONENTS[key]


def require_component_specific_cost(raw_field: str | None) -> SIHCostComponent:
    if raw_field is None or raw_field.strip().lower() in {"cost", "costs", "generic_cost", "hospital_cost", "sih_cost"}:
        raise SIHCostRegistryError("Generic SIH cost request is illegal when component-specific semantics are required.")
    return get_cost_component(raw_field)


def registry_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "registry": "sih_cost_components",
        "components": {key: value.__dict__ for key, value in COST_COMPONENTS.items()},
    }


def cost_entries(*, registry_root: str | Path = "config/registries") -> list[dict[str, Any]]:
    return active_entries(REGISTRY_FILE, registry_root=registry_root)


def cost_component_for_field(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    return match_entry(field, cost_entries(registry_root=registry_root))


def cost_evidence(field: Any, *, registry_root: str | Path = "config/registries") -> dict[str, Any] | None:
    entry = cost_component_for_field(field, registry_root=registry_root)
    if entry is None:
        return None
    return {
        "registry": REGISTRY_FILE,
        "entry_id": entry.get("id"),
        "cost_component": entry.get("cost_component"),
        "protected_non_equivalence": list(entry.get("protected_non_equivalence", []) or []),
    }
'''


def ensure_yaml_entry_descriptions() -> None:
    root = ROOT / "config" / "registries"
    for path in sorted(root.glob("*.yaml")):
        if path.name == "registry_manifest.yaml":
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            continue
        entries = payload.get("entries")
        if not isinstance(entries, list):
            continue
        changed = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "description" not in entry or entry.get("description") in {None, ""}:
                entry["description"] = f"Macro-Slice 27A canonical registry entry {entry.get('id', 'unknown')} from {path.name}."
                changed = True
            if "warnings" not in entry or entry.get("warnings") is None:
                entry["warnings"] = []
                changed = True
        if changed:
            path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")


def regenerate_registry_manifest() -> None:
    root = ROOT / "config" / "registries"
    registries: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("*.yaml")):
        if path.name == "registry_manifest.yaml":
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        registries[path.stem] = {
            "path": path.name,
            "schema_version": str(payload.get("schema_version", "1.0")),
            "registry_version": str(payload.get("registry_version", payload.get("registry_id", "v1.0"))),
            "sha256": sha256_file(path),
        }
    seed = root / "sidra_table_seed.jsonl"
    if seed.exists():
        registries["sidra_table_seed"] = {
            "path": seed.name,
            "schema_version": "jsonl",
            "registry_version": "seed",
            "sha256": sha256_file(seed),
        }
    manifest = {
        "registry_set": {
            "name": "PegaSUS_core",
            "version": "v2.0.slice27a27b",
            "created_at": "2026-06-06",
            "updated_at": "2026-06-13",
            "owner": "local",
            "git_commit": "uninitialized",
            "dirty_allowed": False,
        },
        "registries": registries,
        "entries": [
            {
                "id": "registry_manifest_v2",
                "status": "active",
                "description": "Registry manifest with per-registry content hashes after Slice 27A/27B canonicalization.",
                "warnings": [],
            }
        ],
        "schema_version": "1.0",
        "registry_version": "v2.0",
        "created_at": "2026-06-06",
        "updated_at": "2026-06-13",
        "provenance": "Macro-Slice 27A registry manifest regenerated after registry canonicalization and repair v1.",
    }
    (root / "registry_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )


def patch_tests_and_audit_operator_name() -> None:
    replacements = [
        "tests/unit/test_slice27a27b_registry_canonical_delta.py",
        "scripts/dev/audits/audit_slice27a27b_registry_canonical_delta.py",
    ]
    for rel in replacements:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace('SimpleNamespace(name="RAW")', 'SimpleNamespace(name="raw_field")')
        text = text.replace('operator = SimpleNamespace(name="RAW")', 'operator = SimpleNamespace(name="raw_field")')
        path.write_text(text, encoding="utf-8", newline="\n")


def copy_self() -> None:
    src = Path(__file__).resolve()
    dst = ROOT / "scripts" / "dev" / "updaters" / "repair_slice27a27b_registry_compat_v1.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.copy2(src, dst)


def main() -> None:
    write("src/pegasus/registries/semantic.py", SEMANTIC_MODULE)
    write("src/pegasus/registries/cnes_capacity.py", CNES_CAPACITY_MODULE)
    write("src/pegasus/registries/sih_cost.py", SIH_COST_MODULE)
    ensure_yaml_entry_descriptions()
    regenerate_registry_manifest()
    patch_tests_and_audit_operator_name()
    copy_self()
    print("Applied Slice 27A/27B repair v1: registry compatibility symbols restored, registry entries described, manifest shape restored, and invalid RAW test operator fixed.")


if __name__ == "__main__":
    main()
