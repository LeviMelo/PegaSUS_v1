from __future__ import annotations

import json
from pathlib import Path

import yaml

from pegasus.registries.loader import load_registries
from pegasus.registries.validators import validate_registry_tree


def test_slice27a27b_registry_manifest_hashes_match_written_registries() -> None:
    errors = validate_registry_tree("config/registries")
    assert errors == []
    bundle = load_registries("config/registries")
    manifest = yaml.safe_load(Path("config/registries/registry_manifest.yaml").read_text(encoding="utf-8"))
    entries = manifest.get("registries") or manifest.get("entries") or {}
    for name in ("diagnostic_topology", "cnes_capacity_registry", "sih_cost_registry", "bridge_grammars"):
        assert name in bundle.hashes
        assert name in entries
        assert entries[name]["sha256"] == bundle.hashes[name]


def test_slice27b_legality_source_contains_registry_evidence_hook() -> None:
    source = Path("src/pegasus/efg/legality.py").read_text(encoding="utf-8")
    assert "_append_registry_evidence" in source
    assert "registry_evidence_attached" in source
    assert "diagnostic_topology" in source
    assert "cnes_capacity" in source
    assert "sih_cost" in source
