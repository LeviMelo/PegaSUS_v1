"""Atomic 17-key output-bundle serialization boundary."""

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

PRIMARY_KEYS = {
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


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_variable_dictionary_row(row: dict[str, Any]) -> dict[str, Any]:
    field_id = _clean_text(row.get("field_id") or row.get("id") or "unknown_field")
    display = _clean_text(row.get("display_name") or row.get("name") or field_id)
    technical = _clean_text(row.get("technical_name") or field_id)
    source_systems = row.get("source_systems")
    if source_systems is None:
        source_systems = row.get("source") or row.get("sources") or []
    if not isinstance(source_systems, str):
        source_systems = _json(source_systems)
    return {
        "field_id": field_id,
        "display_name": display,
        "technical_name": technical,
        "definition": _clean_text(
            row.get("definition")
            or row.get("description")
            or f"Compiled epidemiological field {display}."
        ),
        "estimand_label": _clean_text(row.get("estimand_label") or row.get("kind") or "compiled_field"),
        "source_systems": source_systems,
        "carrier": _clean_text(row.get("carrier")),
        "unit": _clean_text(row.get("unit")),
        "support_description": _clean_text(row.get("support_description") or row.get("support_json") or _json(row.get("support") or {})),
        "axis_description": _clean_text(row.get("axis_description") or row.get("axes_json") or _json(row.get("axes") or {})),
        "provenance_description": _clean_text(row.get("provenance_description") or row.get("provenance") or _json([])),
        "state": _clean_text(row.get("state") or "verified"),
        "dashboard_safe": _clean_text(row.get("dashboard_safe") if row.get("dashboard_safe") is not None else "False"),
        "interpretation_warning": _clean_text(row.get("interpretation_warning") or row.get("warnings") or ""),
    }


def _normalize_model_assoc_row(row: dict[str, Any]) -> dict[str, Any]:
    rid = _clean_text(row.get("id") or row.get("model_id") or row.get("residual_id") or row.get("field_id") or "association")
    return {"id": rid, "status": _clean_text(row.get("status") or "recorded"), "warnings": _clean_text(row.get("warnings") or _json([]))}


def _normalize_row(key: str, row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    if key == "VariableDictionary":
        return _normalize_variable_dictionary_row(row)
    if key in {"ModelAssociations", "ResidualAssociations"}:
        return _normalize_model_assoc_row(row)
    return row


def _normalize_rows(key: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalize_row(key, row) for row in rows]


@dataclass
class OutputBundleManager:
    run_dir: Path
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    json_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_dirs: dict[str, Path] = field(default_factory=dict)

    def set_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        if key not in TABLE_KEYS:
            raise ValueError(f"{key!r} is not a first-class table key")
        self.tables[key] = _normalize_rows(key, list(rows))

    def append_table(self, key: str, rows: list[dict[str, Any]]) -> None:
        if key not in TABLE_KEYS:
            raise ValueError(f"{key!r} is not a first-class table key")
        incoming = _normalize_rows(key, list(rows))
        existing = self.tables.get(key, [])
        id_col = PRIMARY_KEYS.get(key)
        if id_col is None:
            self.tables[key] = existing + incoming
            return
        incoming_ids = {str(row[id_col]) for row in incoming if row.get(id_col) is not None}
        kept = [row for row in existing if str(row.get(id_col)) not in incoming_ids]
        self.tables[key] = kept + incoming

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
                self.tables[key] = _normalize_rows(key, pq.read_table(path).to_pylist())
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
        rows = _normalize_rows(key, self.tables.get(key, []))
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
            if item.resolve() == target.resolve():
                continue
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    def write_stage_workspace(self, workspace_dir: str | Path) -> Path:
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


__all__ = ["OutputBundleManager"]
