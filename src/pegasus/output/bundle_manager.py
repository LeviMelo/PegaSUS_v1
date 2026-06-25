"""Atomic 17-key output-bundle serialization boundary.

Production compile stages must stage first-class output surfaces here.
The final run directory is flushed once at Phase E.
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


TABLE_KEYS = {
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

JSON_KEYS = {"UserIntent", "RunConfig", "P_vector", "ReproducibilityManifest"}

DIRECTORY_KEYS = {"Tables", "Maps"}


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

            backup_old = None
            if final.exists():
                backup_old = final.parent / f"{final.name}.pre_phaseE"
                if backup_old.exists():
                    shutil.rmtree(backup_old)
                os.replace(final, backup_old)
            os.replace(tmp, final)
            if backup_old is not None and backup_old.exists():
                shutil.rmtree(backup_old, ignore_errors=True)
            self.run_dir = final
            return final
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
