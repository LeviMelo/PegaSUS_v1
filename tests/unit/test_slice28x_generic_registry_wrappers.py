from __future__ import annotations

from pathlib import Path

import yaml

from pegasus.registries.generic import active_entries, get_entry, registry_manifest


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
