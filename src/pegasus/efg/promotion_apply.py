
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from pegasus.output.table_io import append_replace_rows, read_rows, write_rows_like

import pyarrow as pa

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle


class EFGPromotionApplyError(ValueError):
    """Raised when an EFG promotion plan cannot be safely applied."""


@dataclass(frozen=True)
class EFGPromotionApplyResult:
    run_dir: str
    promotion_plan_path: str
    promoted_field_count: int
    skipped_field_count: int
    blocked_field_count: int
    promoted_field_ids: tuple[str, ...]
    skipped_field_ids: tuple[str, ...]
    blocked_field_ids: tuple[str, ...]
    output_validation_ok: bool

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "gate": "efg_promotion_apply",
            "run_dir": self.run_dir,
            "promotion_plan_path": self.promotion_plan_path,
            "promoted_field_count": self.promoted_field_count,
            "skipped_field_count": self.skipped_field_count,
            "blocked_field_count": self.blocked_field_count,
            "promoted_field_ids": list(self.promoted_field_ids),
            "skipped_field_ids": list(self.skipped_field_ids),
            "blocked_field_ids": list(self.blocked_field_ids),
            "output_validation_ok": self.output_validation_ok,
            "mutates": ["V_fields", "Q_tensor", "VariableDictionary", "Warnings", "QuarantinedFields"],
            "does_not_mutate": ["E_DAG", "ModelAssociations", "ResidualAssociations", "Hypotheses", "ForcedFields"],
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_json(payload), encoding="utf-8")



def _read_rows(path: Path) -> list[dict[str, Any]]:
    return read_rows(path)

def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows_like(path, rows)

def _append_replace(path: Path, new_rows: list[dict[str, Any]], *, id_column: str) -> None:
    append_replace_rows(path, new_rows, id_column=id_column)
def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return [value]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _field_from_promotion(item: dict[str, Any]) -> dict[str, Any]:
    for key in ("field", "field_node", "materialized_field", "v_field"):
        value = item.get(key)
        if isinstance(value, dict):
            return value
    if isinstance(item.get("field_manifest"), dict):
        manifest = item["field_manifest"]
        if isinstance(manifest.get("field"), dict):
            return manifest["field"]
        return manifest
    if "field_id" in item:
        return item
    return {}


def _promotion_items(plan: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "promotions",
        "planned_promotions",
        "eligible_promotions",
        "promotion_candidates",
        "fields",
        "materialized_fields",
    ):
        values = plan.get(key)
        if isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    summary = plan.get("summary")
    if isinstance(summary, dict):
        values = summary.get("promotions")
        if isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    return []


def _is_promotable(item: dict[str, Any], field: dict[str, Any]) -> bool:
    status = str(item.get("status") or item.get("promotion_status") or item.get("decision") or "planned")
    action = str(item.get("action") or item.get("planned_action") or "promote")
    blocked = bool(item.get("blocked") or item.get("conflict") or item.get("has_conflict"))
    materialization_state = str(field.get("materialization_state") or item.get("materialization_state") or "metadata_only")
    return (
        not blocked
        and status in {"planned", "eligible", "ok", "promote", "promotable", "accepted"}
        and action in {"promote", "planned_promote", "append", "apply"}
        and materialization_state == "metadata_only"
    )


def planned_promotion_fields(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for item in _promotion_items(plan):
        field = _field_from_promotion(item)
        field_id = str(field.get("field_id") or item.get("field_id") or "")
        if not field_id:
            skipped.append({"field_id": "", "reason": "missing_field_id", "item": item})
            continue
        if _is_promotable(item, field):
            promoted.append(field)
        elif item.get("blocked") or item.get("conflict") or item.get("has_conflict"):
            blocked.append({"field_id": field_id, "reason": item.get("reason") or "promotion_conflict", "item": item})
        else:
            skipped.append({"field_id": field_id, "reason": item.get("reason") or "not_promotable", "item": item})
    return promoted, skipped, blocked


def _field_value(field: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in field:
        return field[key]
    metadata = field.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return default


def v_field_row(field: dict[str, Any]) -> dict[str, Any]:
    field_id = str(_field_value(field, "field_id"))
    if not field_id or field_id == "None":
        raise EFGPromotionApplyError(f"Cannot promote field without field_id: {field}")
    support = _as_dict(_field_value(field, "support", _field_value(field, "support_json", {})))
    axes = _as_dict(_field_value(field, "axes", _field_value(field, "axes_json", {})))
    role = _as_list(_field_value(field, "role", []))
    source = _as_list(_field_value(field, "source", []))
    provenance = _as_list(_field_value(field, "provenance", []))
    warnings = _as_list(_field_value(field, "warnings", []))
    if "efg_promotion_metadata_only" not in warnings:
        warnings.append("efg_promotion_metadata_only")
    return {
        "field_id": field_id,
        "name": str(_field_value(field, "name", field_id)),
        "kind": str(_field_value(field, "kind", "observer_proxy")),
        "carrier": str(_field_value(field, "carrier", "unknown")),
        "unit": str(_field_value(field, "unit", "unknown")),
        "aggregation": str(_field_value(field, "aggregation", "non_aggregable")),
        "role": _json(role),
        "source": _json(source),
        "support_json": _json(support),
        "axes_json": _json(axes),
        "operator": _field_value(field, "operator"),
        "provenance": _json(provenance),
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": _json(warnings),
        "lineage_hash": str(_field_value(field, "lineage_hash", field_id)),
        "registry_hash": str(_field_value(field, "registry_hash", "efg_promotion_apply_v1")),
        "materialization_state": "metadata_only",
        "path": _field_value(field, "path"),
    }


def q_tensor_row(field: dict[str, Any]) -> dict[str, Any]:
    field_id = str(_field_value(field, "field_id"))
    return {
        "field_id": field_id,
        "n_events": 0.0,
        "n_denom": 0.0,
        "n_eff": 0.0,
        "cov_S": 0.0,
        "cov_T": 0.0,
        "missingness": 1.0,
        "zero_inflation": 0.0,
        "denom_fragility": 1.0,
        "cv": None,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.75,
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": _json(["efg_promotion_metadata_only", "q_tensor_placeholder_no_numerical_tensor"]),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }


def variable_dictionary_row(field: dict[str, Any]) -> dict[str, Any]:
    field_id = str(_field_value(field, "field_id"))
    support = _as_dict(_field_value(field, "support", _field_value(field, "support_json", {})))
    axes = _as_dict(_field_value(field, "axes", _field_value(field, "axes_json", {})))
    source = _as_list(_field_value(field, "source", []))
    provenance = _as_list(_field_value(field, "provenance", []))
    return {
        "field_id": field_id,
        "display_name": str(_field_value(field, "name", field_id)),
        "technical_name": field_id,
        "definition": "Metadata-only EFG field promoted from SHE substrate materialization plan.",
        "estimand_label": "metadata_only_observer_not_model_estimand",
        "source_systems": _json(source),
        "carrier": str(_field_value(field, "carrier", "unknown")),
        "unit": str(_field_value(field, "unit", "unknown")),
        "support_description": _json(support),
        "axis_description": _json(axes),
        "provenance_description": _json(provenance),
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "interpretation_warning": "Metadata-only promoted EFG field; not yet a numerical tensor or model-ready estimand.",
    }


def warning_row(field: dict[str, Any]) -> dict[str, Any]:
    field_id = str(_field_value(field, "field_id"))
    return {
        "warning_id": f"slice15b_efg_promotion_metadata_only_{field_id}",
        "field_id": field_id,
        "source": "pegasus.efg.promotion_apply",
        "severity": "info",
        "code": "efg_promotion_metadata_only",
        "message": "EFG field was promoted as metadata-only; Q tensor row is a non-model placeholder.",
        "inherited_from": _json(_as_list(_field_value(field, "warnings", []))),
        "created_at": _now(),
    }


def quarantined_row(field: dict[str, Any]) -> dict[str, Any]:
    field_id = str(_field_value(field, "field_id"))
    return {
        "field_id": field_id,
        "state": "quarantined_descriptive",
        "reason": "efg_metadata_only_promotion_without_numerical_tensor",
        "warnings": _json(["efg_promotion_metadata_only", "q_tensor_placeholder_no_numerical_tensor"]),
    }


def attach_apply_summary_to_run(run_dir: str | Path, summary: dict[str, Any]) -> None:
    root = Path(run_dir)
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / rel
        if not path.exists():
            continue
        payload = _load_json(path)
        payload["efg_promotion_apply_gate"] = summary
        if rel == "P_vector.json":
            payload.setdefault("provenance", {})
            provenance = payload["provenance"] if isinstance(payload["provenance"], dict) else {}
            for field_id in summary.get("promoted_field_ids", []):
                provenance[str(field_id)] = ["efg_materialization", "metadata_only_promotion", "quarantined_descriptive"]
            payload["provenance"] = provenance
        _write_json(path, payload)


def apply_efg_promotion_plan_to_run(
    *,
    run_dir: str | Path,
    promotion_plan: str | Path | None = None,
    validate: bool = True,
) -> EFGPromotionApplyResult:
    root = Path(run_dir)
    if promotion_plan is None:
        promotion_plan = root / "Tables" / "efg_promotion_plan.json"
    plan_path = Path(promotion_plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"EFG promotion plan not found: {plan_path}")
    missing = [key for key, rel in OUTPUT_BUNDLE_FILES.items() if not (root / rel).exists()]
    if missing:
        raise EFGPromotionApplyError(f"Run directory is not a complete output bundle; missing={missing}")

    plan = _load_json(plan_path)
    fields, skipped, blocked = planned_promotion_fields(plan)
    if not fields:
        summary = EFGPromotionApplyResult(
            run_dir=str(root),
            promotion_plan_path=str(plan_path),
            promoted_field_count=0,
            skipped_field_count=len(skipped),
            blocked_field_count=len(blocked),
            promoted_field_ids=(),
            skipped_field_ids=tuple(str(item.get("field_id", "")) for item in skipped),
            blocked_field_ids=tuple(str(item.get("field_id", "")) for item in blocked),
            output_validation_ok=True,
        )
        attach_apply_summary_to_run(root, summary.as_manifest())
        return summary

    v_rows = [v_field_row(field) for field in fields]
    q_rows = [q_tensor_row(field) for field in fields]
    vd_rows = [variable_dictionary_row(field) for field in fields]
    warning_rows = [warning_row(field) for field in fields]
    quarantined_rows = [quarantined_row(field) for field in fields]

    _append_replace(root / "V_fields.parquet", v_rows, id_column="field_id")
    _append_replace(root / "Q_tensor.parquet", q_rows, id_column="field_id")
    _append_replace(root / "VariableDictionary.parquet", vd_rows, id_column="field_id")
    _append_replace(root / "Warnings.parquet", warning_rows, id_column="warning_id")
    _append_replace(root / "QuarantinedFields.parquet", quarantined_rows, id_column="field_id")

    validation_ok = True
    if validate:
        result = validate_output_bundle(run_dir=str(root))
        validation_ok = bool(result.ok)
        if not result.ok:
            raise EFGPromotionApplyError("Run bundle failed validation after EFG promotion apply: " + "; ".join(result.errors))

    summary = EFGPromotionApplyResult(
        run_dir=str(root),
        promotion_plan_path=str(plan_path),
        promoted_field_count=len(v_rows),
        skipped_field_count=len(skipped),
        blocked_field_count=len(blocked),
        promoted_field_ids=tuple(str(row["field_id"]) for row in v_rows),
        skipped_field_ids=tuple(str(item.get("field_id", "")) for item in skipped),
        blocked_field_ids=tuple(str(item.get("field_id", "")) for item in blocked),
        output_validation_ok=validation_ok,
    )
    attach_apply_summary_to_run(root, summary.as_manifest())
    return summary
