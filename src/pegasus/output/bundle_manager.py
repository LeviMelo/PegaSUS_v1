"""Atomic 17-key output-bundle serialization boundary.

The manager is the Phase-E authority for first-class bundle surfaces. Stages
may stage tables/JSON/artifact directories here. Final publication happens
through flush_to_disk(). write_stage_workspace() is an internal workspace for
legacy numerical routines that still read local parquet inputs; it is not a
final run publication.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.storage import write_table


TABLE_KEYS: set[str] = {
    "V_fields",
    "E_DAG",
    "Q_tensor",
    "Warnings",
    "ModelAssociations",
    "ResidualAssociations",
    "Hypotheses",
    "VariableDictionary",
    "FailedBranches",
    "QuarantinedFields",
    "ForcedFields",
}

JSON_KEYS: set[str] = {
    "UserIntent",
    "RunConfig",
    "P_vector",
    "ReproducibilityManifest",
}

DIRECTORY_KEYS: set[str] = {"Tables", "Maps"}

PRIMARY_KEYS: dict[str, str] = {
    "V_fields": "field_id",
    "E_DAG": "edge_id",
    "Q_tensor": "field_id",
    "Warnings": "warning_id",
    "ModelAssociations": "id",
    "ResidualAssociations": "id",
    "Hypotheses": "hypothesis_id",
    "VariableDictionary": "field_id",
    "FailedBranches": "failed_branch_id",
    "QuarantinedFields": "field_id",
    "ForcedFields": "field_id",
}


@dataclass
class OutputBundleManager:
    run_dir: Path
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    json_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_dirs: dict[str, Path] = field(default_factory=dict)

    def set_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        if key not in TABLE_KEYS:
            raise ValueError(f"{key!r} is not a first-class table key")
        self.tables[key] = list(rows)

    def append_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        if key not in TABLE_KEYS:
            raise ValueError(f"{key!r} is not a first-class table key")
        rows = list(rows)
        existing = self.tables.get(key, [])
        id_col = PRIMARY_KEYS.get(key)
        if id_col is None:
            self.tables[key] = existing + rows
            return
        incoming_ids = {
            str(row[id_col])
            for row in rows
            if id_col in row and row[id_col] is not None
        }
        kept = [
            row for row in existing
            if str(row.get(id_col)) not in incoming_ids
        ]
        self.tables[key] = kept + rows

    def set_json(self, key: str, payload: dict[str, Any]) -> None:
        if key not in JSON_KEYS:
            raise ValueError(f"{key!r} is not a first-class JSON key")
        self.json_payloads[key] = dict(payload)

    def set_artifact_dir(self, key: str, path: str | Path) -> None:
        if key not in DIRECTORY_KEYS:
            raise ValueError(f"{key!r} is not a first-class directory key")
        self.artifact_dirs[key] = Path(path)

    def collect_missing_from_run(self, run_dir: str | Path) -> None:
        root = Path(run_dir)
        for key, rel in OUTPUT_BUNDLE_FILES.items():
            path = root / rel
            if key in self.tables or key in self.json_payloads or key in self.artifact_dirs:
                continue
            if key in TABLE_KEYS and path.exists():
                self.tables[key] = pq.read_table(path).to_pylist()
            elif key in JSON_KEYS and path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self.json_payloads[key] = payload
            elif key in DIRECTORY_KEYS and path.exists():
                self.artifact_dirs[key] = path

    @classmethod
    def from_existing(cls, run_dir: str | Path) -> "OutputBundleManager":
        manager = cls(Path(run_dir))
        manager.collect_missing_from_run(run_dir)
        return manager

    def _write_table_key(self, root: Path, key: str, rel: str) -> None:
        rows = self.tables.get(key, [])
        write_table(root / rel, rows, schema_policy="preserve")

    def _write_json_key(self, root: Path, key: str, rel: str) -> None:
        payload = self.json_payloads.get(key, {})
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _write_dir_key(self, root: Path, key: str, rel: str) -> None:
        dst = root / rel
        dst.mkdir(parents=True, exist_ok=True)
        src = self.artifact_dirs.get(key)
        if src is None or not src.exists():
            return
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)


def write_stage_workspace(self, workspace_dir: str | Path) -> Path:
    """Write current staged first-class surfaces to an internal workspace.

    This is not Phase-E publication. It is only an internal compatibility
    workspace for PIRS/HSIC routines that still read local parquet inputs.

    Critical rule: never copy artifact directory keys (Tables/Maps) from
    the run directory into this workspace. If the workspace is inside or
    near the run tree, copying Tables recursively can create an infinite
    nested copy. The workspace needs first-class tables/JSON plus empty
    Tables/Maps directories for stage artifacts.
    """
    workspace = Path(workspace_dir)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for key, rel in OUTPUT_BUNDLE_FILES.items():
        if key in TABLE_KEYS:
            self._write_table_key(workspace, key, rel)
        elif key in JSON_KEYS:
            self._write_json_key(workspace, key, rel)

    (workspace / "Tables").mkdir(parents=True, exist_ok=True)
    (workspace / "Maps").mkdir(parents=True, exist_ok=True)
    return workspace


    def flush_to_disk(self, run_dir: str | Path | None = None) -> Path:
        final = Path(run_dir) if run_dir is not None else self.run_dir
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"{final.name}.phaseE.", dir=str(final.parent)))
        try:
            for key, rel in OUTPUT_BUNDLE_FILES.items():
                if key in TABLE_KEYS:
                    self._write_table_key(tmp, key, rel)
                elif key in JSON_KEYS:
                    self._write_json_key(tmp, key, rel)
                elif key in DIRECTORY_KEYS:
                    self._write_dir_key(tmp, key, rel)

            old = None
            if final.exists():
                old = final.parent / f"{final.name}.pre_phaseE"
                if old.exists():
                    shutil.rmtree(old)
                os.replace(final, old)
            os.replace(tmp, final)
            if old is not None and old.exists():
                shutil.rmtree(old, ignore_errors=True)
            self.run_dir = final
            return final
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise


__all__ = ["OutputBundleManager", "TABLE_KEYS", "JSON_KEYS", "DIRECTORY_KEYS"]
