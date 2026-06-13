
"""EFG promotion planning for materialized substrate fields.

Slice 15A is intentionally non-mutating with respect to V_fields, E_DAG, and
Q_tensor.  It reads the Slice 14B EFG substrate-materialization manifest and the
existing run bundle, then writes an auditable promotion plan.  Promotion means
"eligible for a future controlled V_fields insertion", not actual insertion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl


class EFGPromotionPlanError(ValueError):
    """Raised when a materialization manifest cannot be promoted safely."""


@dataclass(frozen=True)
class PromotionDecision:
    field_id: str
    name: str
    column: str | None
    carrier: str | None
    unit: str | None
    kind: str | None
    dashboard_safe: str | None
    materialization_state: str | None
    status: str
    reason: str
    source_manifest_index: int | None = None
    field: dict[str, Any] | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "name": self.name,
            "column": self.column,
            "carrier": self.carrier,
            "unit": self.unit,
            "kind": self.kind,
            "dashboard_safe": self.dashboard_safe,
            "materialization_state": self.materialization_state,
            "status": self.status,
            "reason": self.reason,
            "source_manifest_index": self.source_manifest_index,
            "field": dict(self.field) if self.field is not None else None,
        }


@dataclass(frozen=True)
class EFGPromotionPlan:
    materialization_manifest_path: str
    run_dir: str
    planned: tuple[PromotionDecision, ...]
    conflicts: tuple[PromotionDecision, ...]
    excluded_source_fields: tuple[dict[str, Any], ...]
    existing_field_count: int
    existing_field_ids: tuple[str, ...]
    existing_field_names: tuple[str, ...]

    @property
    def planned_count(self) -> int:
        return len(self.planned)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    @property
    def excluded_source_field_count(self) -> int:
        return len(self.excluded_source_fields)

    @property
    def promotion_safe(self) -> bool:
        return self.planned_count > 0 and self.conflict_count == 0

    def summary(self, *, plan_path: str | None = None) -> dict[str, Any]:
        return {
            "status": "planned" if self.promotion_safe else "blocked",
            "promotion_safe": self.promotion_safe,
            "planned_count": self.planned_count,
            "conflict_count": self.conflict_count,
            "excluded_source_field_count": self.excluded_source_field_count,
            "existing_field_count": self.existing_field_count,
            "materialization_manifest_path": self.materialization_manifest_path,
            "promotion_plan_path": plan_path,
            "non_mutating": True,
            "writes_v_fields": False,
            "writes_e_dag": False,
            "writes_q_tensor": False,
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "artifact": "efg_promotion_plan",
            "slice": "15A",
            "materialization_manifest_path": self.materialization_manifest_path,
            "run_dir": self.run_dir,
            "promotion_safe": self.promotion_safe,
            "existing_field_count": self.existing_field_count,
            "existing_field_ids": list(self.existing_field_ids),
            "existing_field_names": list(self.existing_field_names),
            "planned_count": self.planned_count,
            "conflict_count": self.conflict_count,
            "excluded_source_field_count": self.excluded_source_field_count,
            "planned_promotions": [decision.as_manifest() for decision in self.planned],
            "conflicts": [decision.as_manifest() for decision in self.conflicts],
            "excluded_source_fields": list(self.excluded_source_fields),
            "contract": {
                "non_mutating": True,
                "writes_v_fields": False,
                "writes_e_dag": False,
                "writes_q_tensor": False,
                "future_stage": "controlled_v_fields_insertion",
            },
        }


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise EFGPromotionPlanError(f"JSON manifest does not exist: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EFGPromotionPlanError(f"JSON manifest must be an object: {p}")
    return payload


def _iter_nested_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_nested_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_dicts(item)


def _first_list(payload: dict[str, Any], names: tuple[str, ...]) -> list[dict[str, Any]]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for node in _iter_nested_dicts(payload):
        for name in names:
            value = node.get(name)
            if isinstance(value, list):
                items = [item for item in value if isinstance(item, dict)]
                if items:
                    return items
    return []



def materialized_fields_from_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract materialized FieldNode-like rows from Slice 14B manifest shapes.

    Slice 14B serializes materialized entries as wrappers of the form
    ``{"candidate_id": ..., "field": {...}}``.  Promotion planning must
    preserve the full nested FieldNode payload rather than collapsing to the
    small PromotionDecision summary.  Older tests and manual manifests may
    still provide direct field dictionaries; those remain accepted.
    """

    fields = _first_list(
        payload,
        (
            "materialized_fields",
            "field_nodes",
            "fields",
            "nodes",
            "materialized_field_nodes",
        ),
    )
    valid: list[dict[str, Any]] = []
    for item in fields:
        direct = dict(item)
        nested = None
        for key in ("field", "field_node", "materialized_field", "v_field"):
            value = item.get(key)
            if isinstance(value, dict):
                nested = dict(value)
                break
        if nested is not None:
            if item.get("candidate_id") is not None:
                nested.setdefault("candidate_id", item.get("candidate_id"))
            if item.get("lineage_hash") is not None:
                nested.setdefault("lineage_hash", item.get("lineage_hash"))
            if item.get("materialization_reason") is not None:
                nested.setdefault("materialization_reason", item.get("materialization_reason"))
            if "field_id" in nested or "id" in nested or "name" in nested:
                valid.append(nested)
            continue
        if "field_id" in direct or "id" in direct or "name" in direct:
            valid.append(direct)
    return valid

def excluded_fields_from_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _first_list(
        payload,
        (
            "excluded_source_fields",
            "excluded_fields",
            "exclusions",
            "source_exclusions",
        ),
    )


def _support_column(field: dict[str, Any]) -> str | None:
    support = field.get("support")
    if isinstance(support, str):
        try:
            support = json.loads(support)
        except json.JSONDecodeError:
            support = {}
    if isinstance(support, dict):
        value = support.get("column") or support.get("source_column") or support.get("column_name")
        if value is not None:
            return str(value)
    for key in ("column", "source_column", "column_name"):
        value = field.get(key)
        if value is not None:
            return str(value)
    return None


def _json_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _existing_v_fields(run_dir: str | Path) -> tuple[set[str], set[str], int]:
    path = Path(run_dir) / "V_fields.parquet"
    if not path.exists():
        return set(), set(), 0
    df = pl.read_parquet(path)
    ids: set[str] = set()
    names: set[str] = set()
    if "field_id" in df.columns:
        ids.update(str(v) for v in df.get_column("field_id").drop_nulls().to_list())
    if "name" in df.columns:
        names.update(str(v) for v in df.get_column("name").drop_nulls().to_list())
    return ids, names, df.height


def _field_id(field: dict[str, Any], index: int) -> str:
    value = field.get("field_id") or field.get("id")
    if value:
        return str(value)
    column = _support_column(field) or f"field_{index}"
    return f"efg_substrate::{column}"


def _field_name(field: dict[str, Any], field_id: str) -> str:
    value = field.get("name") or field.get("label")
    return str(value) if value is not None else field_id


def _text_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def build_efg_promotion_plan(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path,
) -> EFGPromotionPlan:
    payload = _load_json(materialization_manifest)
    fields = materialized_fields_from_manifest(payload)
    if not fields:
        raise EFGPromotionPlanError("materialization manifest does not contain materialized fields")

    excluded = tuple(dict(item, efg_materialized=False) for item in excluded_fields_from_manifest(payload))
    existing_ids, existing_names, existing_count = _existing_v_fields(run_dir)

    planned: list[PromotionDecision] = []
    conflicts: list[PromotionDecision] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    for index, field in enumerate(fields):
        field_id = _field_id(field, index)
        name = _field_name(field, field_id)
        column = _support_column(field)
        support = _json_cell(field.get("support"))
        carrier = _text_or_none(field.get("carrier"))
        unit = _text_or_none(field.get("unit"))
        kind = _text_or_none(field.get("kind"))
        dashboard_safe = _text_or_none(field.get("dashboard_safe"))
        materialization_state = _text_or_none(field.get("materialization_state"))

        reason = "eligible_for_future_v_fields_insertion"
        status = "planned"
        if field_id in existing_ids:
            status = "conflict"
            reason = "field_id_already_present_in_v_fields"
        elif name in existing_names:
            status = "conflict"
            reason = "field_name_already_present_in_v_fields"
        elif field_id in seen_ids:
            status = "conflict"
            reason = "duplicate_materialized_field_id"
        elif name in seen_names:
            status = "conflict"
            reason = "duplicate_materialized_field_name"
        elif materialization_state and materialization_state != "metadata_only":
            status = "conflict"
            reason = f"unsupported_materialization_state:{materialization_state}"
        elif support and support.get("efg_materialized") is False:
            status = "conflict"
            reason = "field_support_marks_efg_materialized_false"

        decision = PromotionDecision(
            field_id=field_id,
            name=name,
            column=column,
            carrier=carrier,
            unit=unit,
            kind=kind,
            dashboard_safe=dashboard_safe,
            materialization_state=materialization_state,
            status=status,
            reason=reason,
            source_manifest_index=index,
            field=dict(field),
        )
        if status == "planned":
            planned.append(decision)
        else:
            conflicts.append(decision)
        seen_ids.add(field_id)
        seen_names.add(name)

    return EFGPromotionPlan(
        materialization_manifest_path=str(Path(materialization_manifest)),
        run_dir=str(Path(run_dir)),
        planned=tuple(planned),
        conflicts=tuple(conflicts),
        excluded_source_fields=excluded,
        existing_field_count=existing_count,
        existing_field_ids=tuple(sorted(existing_ids)),
        existing_field_names=tuple(sorted(existing_names)),
    )


def write_efg_promotion_plan(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    plan = build_efg_promotion_plan(run_dir=run_dir, materialization_manifest=materialization_manifest)
    output_path = Path(output) if output is not None else Path(run_dir) / "Tables" / "efg_promotion_plan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.as_manifest(), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return {
        "plan": plan.as_manifest(),
        "summary": plan.summary(plan_path=str(output_path)),
        "path": str(output_path),
    }


def _update_json(path: Path, key: str, value: dict[str, Any]) -> None:
    payload: dict[str, Any]
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}
    payload[key] = value
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def attach_efg_promotion_plan_to_run(
    *,
    run_dir: str | Path,
    materialization_manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = Path(materialization_manifest) if materialization_manifest is not None else root / "Tables" / "efg_substrate_materialization.json"
    result = write_efg_promotion_plan(run_dir=root, materialization_manifest=manifest_path)
    summary = dict(result["summary"])
    for relative in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        _update_json(root / relative, "efg_promotion_gate", summary)
    return summary
