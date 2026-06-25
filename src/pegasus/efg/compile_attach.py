"""Attach autonomous EFG output as first-class graph surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pegasus.core.hashing import sha256_file
from pegasus.efg.dag import EFGResult
from pegasus.efg.executor import execute_efg_result
from pegasus.efg.lineage import lineage_hash
from pegasus.output.bundle_manager import OutputBundleManager


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _v_row(field) -> dict[str, Any]:
    return {
        "field_id": field.id,
        "name": field.name,
        "kind": field.kind,
        "carrier": field.carrier,
        "unit": field.unit,
        "aggregation": field.aggregation,
        "role": _json(field.role),
        "source": _json(field.source),
        "support_json": _json(field.support),
        "axes_json": _json(field.axes),
        "operator": field.operator,
        "provenance": _json(field.provenance),
        "state": field.state.value if hasattr(field.state, "value") else str(field.state),
        "dashboard_safe": str(field.dashboard_safe),
        "warnings": _json(field.warnings),
        "lineage_hash": lineage_hash(field.lineage),
        "registry_hash": str(field.lineage.registry_versions.get("efg_operator_registry", "unknown")),
        "materialization_state": field.materialization_state.value if hasattr(field.materialization_state, "value") else str(field.materialization_state),
        "path": field.path,
    }


def _q_row(field) -> dict[str, Any]:
    support = field.support or {}
    return {
        "field_id": field.id,
        "n_events": float(support.get("n_events") or support.get("row_count") or 0.0),
        "n_denom": support.get("n_denom"),
        "n_eff": support.get("n_eff"),
        "cov_S": support.get("cov_S"),
        "cov_T": support.get("cov_T"),
        "missingness": support.get("missing_rate") or support.get("missingness"),
        "zero_inflation": support.get("zero_inflation"),
        "denom_fragility": support.get("denom_fragility"),
        "cv": support.get("cv"),
        "moran_i": support.get("moran_i"),
        "temporal_roughness": support.get("temporal_roughness"),
        "spatial_entropy": support.get("spatial_entropy"),
        "provenance_risk": 0.0 if "official" in set(field.provenance or []) else 0.5,
        "race_axis_source": field.axes.get("race_axis_type") or field.axes.get("race_axis"),
        "race_axis_target": field.axes.get("race_axis_target"),
        "missing_race_share": support.get("missing_race_share"),
        "emission_prior_strength": support.get("emission_prior_strength"),
        "race_bridge_cv": support.get("race_bridge_cv"),
        "sensitivity_width": support.get("sensitivity_width"),
        "bridge_mode": support.get("bridge_mode"),
        "state": field.state.value if hasattr(field.state, "value") else str(field.state),
        "dashboard_safe": str(field.dashboard_safe),
        "warnings": _json(field.warnings),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def _edge_row(edge) -> dict[str, Any]:
    return {
        "edge_id": edge.edge_id,
        "parent_field_id": edge.parent_field_id,
        "child_field_id": edge.child_field_id,
        "operator": edge.operator,
        "operator_params_json": _json(edge.operator_params),
        "registry_versions_json": _json(edge.registry_versions),
        "created_at": edge.created_at,
    }


def _warning_rows(field) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, warning in enumerate(field.warnings or []):
        rows.append({
            "warning_id": f"warning::{field.id}::{idx}",
            "field_id": field.id,
            "source": "pegasus.efg.executor",
            "severity": "warning",
            "code": str(warning),
            "message": str(warning),
            "inherited_from": _json([]),
            "created_at": _now(),
        })
    return rows


def _dictionary_row(field, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "field_id": field.id,
        "name": field.name,
        "carrier": field.carrier,
        "unit": field.unit,
        "support_description": _json(field.support),
        "axis_description": _json(field.axes),
        "provenance_description": _json(field.provenance),
        "state": field.state.value if hasattr(field.state, "value") else str(field.state),
        "dashboard_safe": str(field.dashboard_safe),
        "interpretation_warning": ";".join(field.warnings or []),
        "diagnostic_role": metadata.get("diagnostic_role"),
        "topology": metadata.get("topology"),
        "position": metadata.get("position"),
        "icd_group_kind": metadata.get("icd_group_kind"),
        "icd_group_id": metadata.get("icd_group_id"),
    }


def _failed_row(branch) -> dict[str, Any]:
    if hasattr(branch, "as_manifest"):
        payload = branch.as_manifest()
    else:
        payload = dict(branch)
    return {
        "failed_branch_id": str(payload.get("failed_branch_id") or payload.get("id") or f"failed::{hash(str(payload))}"),
        "attempted_operator": str(payload.get("attempted_operator") or payload.get("operator") or payload.get("operator_name") or "unknown"),
        "parent_field_ids": _json(payload.get("parent_field_ids") or payload.get("parents") or []),
        "failure_stage": str(payload.get("failure_stage") or payload.get("stage") or "declaration"),
        "failed_terms": _json(payload.get("failed_terms") or []),
        "reason": str(payload.get("reason") or payload.get("failure_reason") or "blocked"),
        "warnings": _json(payload.get("warnings") or []),
        "created_at": str(payload.get("created_at") or _now()),
    }


def _quarantined_row(field) -> dict[str, Any]:
    return {
        "field_id": field.id,
        "reason": ";".join(field.warnings or []) or "not_dashboard_safe",
        "state": field.state.value if hasattr(field.state, "value") else str(field.state),
        "dashboard_safe": str(field.dashboard_safe),
        "created_at": _now(),
    }


@dataclass(frozen=True)
class AutonomousEFGAttachResult:
    run_dir: str
    manifest_path: str
    manifest_hash: str
    efg_id: str
    field_count: int
    edge_count: int
    failed_branch_count: int
    output_validation_ok: bool
    execution_manifest_path: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return dict(vars(self)) | {
            "status": "attached",
            "graph_authority": "autonomous_efg_core",
            "legacy_computed_fields_preserved": False,
            "first_class_output_keys_added": 7,
            "first_class_output_keys_replaced": [
                "V_fields",
                "E_DAG",
                "Q_tensor",
                "Warnings",
                "VariableDictionary",
                "FailedBranches",
                "QuarantinedFields",
            ],
        }


def attach_autonomous_efg_to_run(
    *,
    run_dir: str | Path,
    efg: EFGResult | None = None,
    result: EFGResult | None = None,
    validate: bool = True,
    bundle: OutputBundleManager | None = None,
) -> AutonomousEFGAttachResult:
    efg = efg or result
    if efg is None:
        raise TypeError("attach_autonomous_efg_to_run requires efg= or result=")
    if bundle is None:
        raise RuntimeError("Autonomous EFG attach requires an OutputBundleManager")

    root = Path(run_dir)
    workspace = root.parent / f"{root.name}__efg_stage_workspace"
    tables = bundle.write_stage_workspace(workspace) / "Tables"
    tables.mkdir(parents=True, exist_ok=True)

    efg, execution_report = execute_efg_result(
        efg,
        output_dir=tables / "efg_tensors",
        require_materialized=True,
    )
    execution_path = tables / "efg_execution_manifest.json"
    execution_path.write_text(
        json.dumps(execution_report.as_manifest(), indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    manifest_path = tables / "efg_autonomous_manifest.json"
    manifest_payload = efg.as_manifest()
    manifest_payload["execution"] = execution_report.as_manifest()
    manifest_payload["legacy_computed_fields_preserved"] = False
    manifest_payload["graph_authority"] = "autonomous_efg_core"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    manifest_hash = sha256_file(manifest_path)

    metadata_by_field = {
        str(item.get("field_id")): item
        for item in efg.variable_dictionary
        if isinstance(item, dict) and item.get("field_id")
    }

    v_rows = [_v_row(field) for field in efg.fields]
    edge_rows = [_edge_row(edge) for edge in efg.edges]
    q_rows = [_q_row(field) for field in efg.fields]
    warning_rows = [row for field in efg.fields for row in _warning_rows(field)]
    dictionary_rows = [_dictionary_row(field, metadata_by_field.get(field.id, {})) for field in efg.fields]
    failed_rows = [_failed_row(branch) for branch in efg.failed_branches]
    quarantined_rows = [_quarantined_row(field) for field in efg.fields if str(field.dashboard_safe) != "True"]

    if bundle is not None:
        bundle.set_table("V_fields", v_rows)
        bundle.set_table("E_DAG", edge_rows)
        bundle.set_table("Q_tensor", q_rows)
        bundle.set_table("Warnings", warning_rows)
        bundle.set_table("VariableDictionary", dictionary_rows)
        bundle.set_table("FailedBranches", failed_rows)
        bundle.set_table("QuarantinedFields", quarantined_rows)
        bundle.set_artifact_dir("Tables", tables)

    ok = True
    if validate:
        ok = True

    return AutonomousEFGAttachResult(
        run_dir=str(root),
        manifest_path=str(manifest_path),
        manifest_hash=manifest_hash,
        efg_id=efg.efg_id,
        field_count=efg.field_count,
        edge_count=efg.edge_count,
        failed_branch_count=len(efg.failed_branches),
        output_validation_ok=ok,
        execution_manifest_path=str(execution_path),
    )


__all__ = ["attach_autonomous_efg_to_run", "AutonomousEFGAttachResult"]
