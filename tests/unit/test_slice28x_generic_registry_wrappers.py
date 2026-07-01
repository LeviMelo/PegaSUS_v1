from __future__ import annotations

from pathlib import Path

import yaml

from pegasus.registries.generic import active_entries, get_entry, registry_manifest
from pegasus.registries.models import active_model_entries
from pegasus.registries.nulls import active_null_entries


def _write_registry(root: Path, name: str) -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
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
