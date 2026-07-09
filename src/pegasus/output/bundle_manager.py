"""Atomic 17-key output-bundle serialization boundary."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pegasus.core.io_utils import atomic_replace
from pegasus.output.schemas import INFERENCE_KEYS, OUTPUT_BUNDLE_FILES, required_nonempty_keys
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
# PROFILE_NONEMPTY / required_nonempty_keys are the single-source-of-truth output
# contract in output.schemas (shared with the validator so the two never drift).

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
    out = dict(row)
    out["id"] = rid
    out["status"] = _clean_text(out.get("status") or "recorded")
    out["warnings"] = _clean_text(out.get("warnings") or _json([]))
    for key in (
        "covariate_field_id",
        "covariate_field_ids",
        "diagnostics_json",
        "warnings",
    ):
        value = out.get(key)
        if value is not None and not isinstance(value, str):
            out[key] = _json(value)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        # Rows in self.tables are already normalized at ingest (set_table /
        # append_table / collect_missing_from_run), and normalization is
        # idempotent, so re-normalizing here would be an O(rows) Python rebuild
        # of the whole table on every flush (and every table flushes twice:
        # write_stage_workspace + flush_to_disk). Write the stored rows directly.
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
            if item.resolve() == target.resolve():
                continue
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    def _artifact_nonempty_in_manager(self, key: str) -> bool:
        if key in TABLE_KEYS:
            return bool(self.tables.get(key))
        if key in JSON_KEYS:
            return bool(self.json_payloads.get(key))
        if key in DIRECTORY_KEYS:
            src = self.artifact_dirs.get(key)
            return src is not None and src.exists() and any(src.iterdir())
        return False

    def _run_profile(self) -> str:
        for key in ("UserIntent", "RunConfig", "ReproducibilityManifest"):
            profile = self.json_payloads.get(key, {}).get("run_profile")
            if profile:
                return str(profile)
        return "core_vital"

    def _execution_stage(self) -> str:
        for key in ("UserIntent", "RunConfig", "ReproducibilityManifest"):
            stage = self.json_payloads.get(key, {}).get("execution_stage")
            if stage:
                return str(stage)
        return "investigate"

    def _append_empty_by_profile_warnings(self) -> None:
        # Emit a declared-empty row for every empty first-class key that is NOT
        # required non-empty for this (run_profile, execution_stage) -- so the
        # anti-silence contract holds and the validator (which shares
        # required_nonempty_keys) never flags a silently-empty artifact. Inference
        # outputs empty below `investigate` are tagged empty_by_stage; everything
        # else empty_by_profile. MSD-II §II.5 / MII-SCOPE-01 / MII-OUT-01.
        run_profile = self._run_profile()
        execution_stage = self._execution_stage()
        required = required_nonempty_keys(run_profile, execution_stage)
        existing = self.tables.get("Warnings", [])
        existing_ids = {str(row.get("warning_id")) for row in existing}
        rows = list(existing)
        for key in OUTPUT_BUNDLE_FILES:
            if key == "Warnings" or key in required or self._artifact_nonempty_in_manager(key):
                continue
            if execution_stage != "investigate" and key in INFERENCE_KEYS:
                warning_id = f"empty_by_stage::{execution_stage}::{key}"
                code = "empty_by_stage"
                message = f"{key} is empty because execution_stage={execution_stage} does not run inference (produced at investigate)."
            else:
                warning_id = f"empty_by_profile::{run_profile}::{key}"
                code = "empty_by_profile"
                message = f"{key} is empty because run_profile={run_profile} does not require it."
            if warning_id in existing_ids:
                continue
            rows.append({
                "warning_id": warning_id,
                "field_id": "run",
                "source": "output_profile",
                "severity": "info",
                "code": code,
                "message": message,
                "inherited_from": "[]",
                "created_at": _now(),
            })
        self.tables["Warnings"] = rows

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
            elif key in DIRECTORY_KEYS:
                self._write_dir_key(workspace, key, rel)
        (workspace / "Tables").mkdir(parents=True, exist_ok=True)
        (workspace / "Maps").mkdir(parents=True, exist_ok=True)
        return workspace

    def _check_first_class_consistency(self) -> None:
        """Surface (never silently emit) a structurally-inconsistent bundle: an E_DAG edge whose
        parent/child references a field id absent from V_fields — a partial/corrupt-bundle symptom.
        Warns rather than raising so a legitimate edge case cannot break the atomic flush."""
        v_rows = self.tables.get("V_fields") or []
        e_rows = self.tables.get("E_DAG") or []
        if not v_rows or not e_rows:
            return
        field_ids = {str(r.get("field_id")) for r in v_rows if r.get("field_id") is not None}
        if not field_ids:
            return
        orphans = {
            str(r.get(k)) for r in e_rows for k in ("parent_field_id", "child_field_id")
            if r.get(k) is not None and str(r.get(k)) not in field_ids
        }
        if orphans:
            warnings.warn(
                f"Output bundle consistency: {len(orphans)} E_DAG edge endpoint(s) reference field ids "
                f"absent from V_fields (e.g. {sorted(orphans)[:3]}) — possible partial/corrupt bundle.",
                RuntimeWarning, stacklevel=2,
            )

    def flush_to_disk(self, run_dir: str | Path | None = None) -> Path:
        final = Path(run_dir) if run_dir is not None else self.run_dir
        final.parent.mkdir(parents=True, exist_ok=True)
        self._append_empty_by_profile_warnings()
        self._check_first_class_consistency()
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
                atomic_replace(final, old)
            atomic_replace(tmp, final)
            if old is not None and old.exists():
                shutil.rmtree(old, ignore_errors=True)
            self.run_dir = final
            return final
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise


__all__ = ["OutputBundleManager"]
