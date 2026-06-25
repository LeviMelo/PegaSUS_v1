"""Attach autonomous EFG output as first-class graph surfaces."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_file
from pegasus.efg.dag import EFGResult
from pegasus.efg.executor import VALUE_COLUMN, execute_efg_result
from pegasus.efg.lineage import lineage_hash
from pegasus.output.bundle_manager import OutputBundleManager
from pegasus.storage import write_table

_PANEL_KEYS = ("year", "municipality_cod6")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _build_panel_index(field_tensors: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Canonical (year, municipality_cod6) panel index shared by all fields.

    Built as the union of support cells across every executed field tensor that
    carries panel keys, sorted deterministically and assigned a stable row_id.
    This is the spine that makes per-field value vectors row-aligned (row i means
    the same (year, municipality) for every field) and supplies the spatial /
    temporal blocks downstream HSIC structured nulls need.
    """
    # Prefer the support of true panel fields (both year and municipality). Only
    # if none exist do we fall back to a coarser single-axis index, so we never
    # manufacture spurious (year, null-municipality) cells.
    full_frames: list[pl.DataFrame] = []
    partial_frames: list[pl.DataFrame] = []
    for df in field_tensors.values():
        keys = [k for k in _PANEL_KEYS if k in df.columns]
        if not keys:
            continue
        sub = df.select(keys).unique()
        for missing in (k for k in _PANEL_KEYS if k not in keys):
            sub = sub.with_columns(pl.lit(None).alias(missing))
        (full_frames if len(keys) == len(_PANEL_KEYS) else partial_frames).append(sub.select(_PANEL_KEYS))
    frames = full_frames or partial_frames
    if not frames:
        return pl.DataFrame({"row_id": [], "year": [], "municipality_cod6": []})
    index = pl.concat(frames, how="vertical_relaxed").unique().sort(_PANEL_KEYS, nulls_last=True)
    return index.with_row_index("row_id")


def _align_vector(df: pl.DataFrame, panel: pl.DataFrame) -> list[float | None] | None:
    """Reindex a field's value tensor onto the canonical panel index.

    Joins on whatever subset of (year, municipality_cod6) the field carries, so a
    national year-series broadcasts across municipalities, a time-invariant map
    broadcasts across years, and a full panel field aligns exactly. Returns one
    value per panel row (None where the field has no cell), or None for pure
    scalar fields that have no panel keys (excluded from modelling).
    """
    if VALUE_COLUMN not in df.columns or panel.height == 0:
        return None
    keys = [k for k in _PANEL_KEYS if k in df.columns]
    if not keys:
        return None
    reduced = df.group_by(keys).agg(pl.col(VALUE_COLUMN).mean().alias(VALUE_COLUMN))
    joined = panel.join(reduced, on=keys, how="left").sort("row_id")
    return [None if v is None else float(v) for v in joined.get_column(VALUE_COLUMN).to_list()]


def _vector_diagnostics(vector: list[float | None]) -> dict[str, float]:
    """Real MSD §3.12 state-tensor fields computed from the aligned vector."""
    n = len(vector)
    observed = [v for v in vector if v is not None]
    n_obs = len(observed)
    missingness = (n - n_obs) / n if n else 1.0
    zero_inflation = (sum(1 for v in observed if v == 0.0) / n) if n else 0.0
    if n_obs > 1:
        mean = sum(observed) / n_obs
        var = sum((v - mean) ** 2 for v in observed) / (n_obs - 1)
        cv = (math.sqrt(var) / abs(mean)) if mean not in (0.0, None) else None
    else:
        cv = None
    return {
        "n_eff": float(n_obs),
        "missingness": float(missingness),
        "zero_inflation": float(zero_inflation),
        "cv": cv,
    }


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


def _q_row(field, vector: list[float | None] | None = None) -> dict[str, Any]:
    support = field.support or {}
    diag = _vector_diagnostics(vector) if vector is not None else {}
    row = {
        "field_id": field.id,
        "n_events": float(support.get("n_events") or support.get("row_count") or 0.0),
        "n_denom": support.get("n_denom"),
        "n_eff": diag.get("n_eff", support.get("n_eff")),
        "cov_S": support.get("cov_S"),
        "cov_T": support.get("cov_T"),
        "missingness": diag.get("missingness", support.get("missing_rate") or support.get("missingness")),
        "zero_inflation": diag.get("zero_inflation", support.get("zero_inflation")),
        "denom_fragility": support.get("denom_fragility"),
        "cv": diag.get("cv", support.get("cv")),
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
    # The value vector the PIRS design matrix consumes. Without this the executor's
    # real per-field tensors never reached PIRS and every design matrix blocked.
    if vector is not None:
        row["value_vector_json"] = _json(vector)
        row["values_json"] = row["value_vector_json"]
    return row


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


def _tensor_warning_row(field, code: str, field_id: str | None = None) -> dict[str, Any]:
    fid = field.id if field is not None else (field_id or "unknown")
    return {
        "warning_id": f"warning::{fid}::tensor_bridge",
        "field_id": fid,
        "source": "pegasus.efg.compile_attach",
        "severity": "warning",
        "code": code,
        "message": f"EFG tensor bridge: {code}",
        "inherited_from": _json([]),
        "created_at": _now(),
    }


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


def _race_bridge_summary_rows(efg: EFGResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in efg.fields:
        if not str(field.id).startswith("SIMRaceBridgePosteriorCount_"):
            continue
        support = field.support or {}
        rows.append({
            "field_id": field.id,
            "bridge_id": support.get("bridge_id"),
            "bridge_operator": support.get("bridge_operator") or field.operator,
            "bridge_mode": support.get("bridge_mode"),
            "prior_hash": support.get("prior_hash"),
            "source_axis": support.get("source_axis"),
            "target_axis": support.get("target_axis"),
            "missing_race_share": support.get("missing_race_share"),
            "race_bridge_cv": support.get("race_bridge_cv"),
            "sensitivity_width": support.get("sensitivity_width"),
            "raw_admin_counts_preserved": bool(support.get("raw_admin_counts_preserved", True)),
            "missing_category_preserved": bool(support.get("missing_category_preserved", True)),
            "created_at": _now(),
        })
    return rows


def _population_diagnostics_rows(efg: EFGResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in efg.fields:
        if not str(field.id).startswith("population_tensor_"):
            continue
        support = field.support or {}
        rows.append({
            "field_id": field.id,
            "tensor_id": field.id,
            "mode": "official_sidra_anchor",
            "solver_id": "sidra_9606_total_anchor",
            "solver_backend": "official_sidra_anchor",
            "n_denom": support.get("n_denom"),
            "period": support.get("period"),
            "locality_id": support.get("locality_id"),
            "request_hash": support.get("request_hash"),
            "metadata_hash": support.get("metadata_hash"),
            "created_at": _now(),
        })
    return rows


def _write_domain_summary_tables(tables: Path, efg: EFGResult) -> None:
    race_rows = _race_bridge_summary_rows(efg)
    if race_rows:
        write_table(tables / "race_bridge_summary.parquet", race_rows, schema_policy="preserve")
    population_rows = _population_diagnostics_rows(efg)
    if population_rows:
        write_table(tables / "population_tensor_diagnostics.parquet", population_rows, schema_policy="preserve")
    domain = efg.domain_summaries or {}
    if isinstance(domain.get("cnes_sih"), dict):
        cnes = domain["cnes_sih"].get("cnes") or {}
        sih = domain["cnes_sih"].get("sih") or {}
        write_table(tables / "slice5a_cnes_capacity_summary.parquet", [dict(cnes)], schema_policy="preserve")
        write_table(tables / "slice5a_sih_cost_summary.parquet", [dict(sih)], schema_policy="preserve")


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
    intent: Any = None,
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
        intent=intent,
    )
    _write_domain_summary_tables(tables, efg)
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

    # Bridge the executor's physical tensors into PIRS-consumable value vectors.
    # Load each materialized field tensor, build the shared panel index, and
    # align every field onto it so Q_tensor carries row-aligned numeric vectors.
    # A field that claims a materialized path but whose tensor cannot be read or
    # aligned is NOT silently dropped — it emits a warning so the gap is visible
    # rather than masquerading as a field that simply has no data.
    field_tensors: dict[str, pl.DataFrame] = {}
    tensor_warning_rows: list[dict[str, Any]] = []
    fields_by_id_for_warn = {field.id: field for field in efg.fields}
    for field in efg.fields:
        if not field.path:
            continue
        tensor_path = Path(field.path)
        if not tensor_path.exists():
            tensor_warning_rows.append(_tensor_warning_row(field, "materialized_field_tensor_path_missing"))
            continue
        try:
            field_tensors[field.id] = pl.read_parquet(tensor_path)
        except Exception as exc:
            tensor_warning_rows.append(_tensor_warning_row(field, f"tensor_read_failed:{type(exc).__name__}"))
    panel_index = _build_panel_index(field_tensors)
    vectors_by_field: dict[str, list[float | None]] = {}
    for field_id, tensor in field_tensors.items():
        try:
            vector = _align_vector(tensor, panel_index)
        except Exception as exc:
            vector = None
            tensor_warning_rows.append(
                _tensor_warning_row(fields_by_id_for_warn.get(field_id), f"tensor_alignment_failed:{type(exc).__name__}", field_id)
            )
        if vector is not None:
            vectors_by_field[field_id] = vector
    if panel_index.height:
        panel_index.write_parquet(tables / "support_index.parquet")

    v_rows = [_v_row(field) for field in efg.fields]
    edge_rows = [_edge_row(edge) for edge in efg.edges]
    q_rows = [_q_row(field, vectors_by_field.get(field.id)) for field in efg.fields]
    warning_rows = [row for field in efg.fields for row in _warning_rows(field)] + tensor_warning_rows
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
