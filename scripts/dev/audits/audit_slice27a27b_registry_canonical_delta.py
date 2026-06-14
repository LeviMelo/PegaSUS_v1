from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from pegasus.core.hashing import sha256_file
from pegasus.efg.legality import evaluate_delta
from pegasus.registries.cnes_capacity import capacity_evidence
from pegasus.registries.diagnostic_topology import diagnostic_evidence
from pegasus.registries.sih_cost import cost_evidence


CRITICAL = [
    "diagnostic_topology.yaml",
    "cnes_capacity_registry.yaml",
    "sih_cost_registry.yaml",
    "clinical_event_definitions.yaml",
    "bridge_grammars.yaml",
    "race_axis_registry.yaml",
    "icd_catalog.yaml",
    "icd_quality_groups.yaml",
    "join_affordances.yaml",
    "municipality_crosswalk_sources.yaml",
    "model_registry.yaml",
    "residual_registry.yaml",
]


def _state(value: str = "active"):
    return SimpleNamespace(value=value)


def _field(**kwargs):
    payload = {
        "field_id": "field",
        "name": "field",
        "carrier": "Deaths",
        "unit": "counts",
        "aggregation": "additive",
        "state": _state(),
        "role_json": "[]",
        "source_json": "[]",
        "metadata_json": "{}",
        "operator": "raw",
    }
    payload.update(kwargs)
    return SimpleNamespace(**payload)


def main() -> int:
    root = Path("config/registries")
    errors: list[str] = []
    manifest = yaml.safe_load((root / "registry_manifest.yaml").read_text(encoding="utf-8"))
    entries = manifest.get("registries") or manifest.get("entries") or {}
    for filename in CRITICAL:
        path = root / filename
        if not path.exists():
            errors.append(f"missing registry: {filename}")
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload.get("registry_version") != "v2.0":
            errors.append(f"registry not v2.0: {filename}")
        registry_entries = payload.get("entries") or []
        if not registry_entries:
            errors.append(f"registry has no entries: {filename}")
        for entry in registry_entries:
            warnings = set(entry.get("warnings") or [])
            if entry.get("status") == "deferred" and "scaffold_only" in warnings:
                errors.append(f"registry still scaffold-only: {filename}:{entry.get('id')}")
        stem = filename.removesuffix(".yaml")
        if stem not in entries:
            errors.append(f"registry missing from manifest: {stem}")
        elif entries[stem].get("sha256") != sha256_file(path):
            errors.append(f"registry manifest hash mismatch: {stem}")

    diagnostic = _field(field_id="sim_CAUSABAS", name="CAUSABAS", unit="ICD10", aggregation="non_aggregable")
    cnes = _field(field_id="cnes_QTLEIT", name="QTLEIT bed capacity", carrier="Facilities", unit="beds")
    sih = _field(field_id="sih_VAL_SP", name="VAL_SP professional service cost", carrier="HospitalAdmissions", unit="BRL")
    if not diagnostic_evidence(diagnostic):
        errors.append("diagnostic evidence helper failed")
    if not capacity_evidence(cnes):
        errors.append("CNES capacity evidence helper failed")
    if not cost_evidence(sih):
        errors.append("SIH cost evidence helper failed")
    delta = evaluate_delta(parents=[diagnostic], operator=SimpleNamespace(name="raw_field"))
    if "registry_evidence_attached" not in set(delta.warnings):
        errors.append("DeltaResult warnings missing registry evidence marker")

    payload = {"slice": "27A/27B", "errors": errors, "critical_registries": CRITICAL}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
