from __future__ import annotations

import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "15B"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/efg/promotion_apply.py",
    "src/pegasus/workflows/efg_apply.py",
    "tests/unit/test_slice15b_efg_promotion_apply.py",
    "tests/integration/test_slice15b_apply_efg_promotions_to_bundle.py",
    "scripts/dev/audits/audit_slice15b_efg_promotion_apply.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/source_registry.py",
    "src/pegasus/she/substrate.py",
    "src/pegasus/pirs/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice15b] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if path.startswith(FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def preflight() -> None:
    required = (
        "src/pegasus/efg/materialize.py",
        "src/pegasus/efg/materialization_manifest.py",
        "src/pegasus/efg/promotion_plan.py",
        "src/pegasus/workflows/efg_promotion.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/bundle.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if compile_text.count("def run_compile(") != 1 or "def _run_compile_impl(" not in compile_text:
        fail("compile.py public/private run_compile boundary is not in the Slice 13C shape")
    if "def attach_efg_promotion_plan_to_run" not in read("src/pegasus/efg/promotion_plan.py"):
        fail("Slice 15A promotion plan API not found")


def write_promotion_apply_module() -> None:
    code = r'''
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

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
    table = pq.read_table(path)
    return table.to_pylist()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_schema(path)
    normalized = [{name: row.get(name) for name in schema.names} for row in rows]
    table = pa.Table.from_pylist(normalized, schema=schema)
    pq.write_table(table, path)


def _append_replace(path: Path, new_rows: list[dict[str, Any]], *, id_column: str) -> None:
    if not new_rows:
        return
    existing = _read_rows(path)
    ids = {str(row[id_column]) for row in new_rows if row.get(id_column) is not None}
    kept = [row for row in existing if str(row.get(id_column)) not in ids]
    _write_rows(path, kept + new_rows)


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
'''
    write("src/pegasus/efg/promotion_apply.py", code)


def write_workflow_module() -> None:
    code = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run


def run_apply_efg_promotion_plan(
    *,
    run_dir: str | Path,
    promotion_plan: str | Path | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    result = apply_efg_promotion_plan_to_run(
        run_dir=run_dir,
        promotion_plan=promotion_plan,
        validate=validate,
    )
    return result.as_manifest()
'''
    write("src/pegasus/workflows/efg_apply.py", code)


def write_tests() -> None:
    unit = r'''
from __future__ import annotations

from pegasus.efg.promotion_apply import planned_promotion_fields, q_tensor_row, v_field_row, variable_dictionary_row


def _field() -> dict:
    return {
        "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
        "name": "SIM-DO underlying ICD observer",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "ICD10",
        "aggregation": "non_aggregable",
        "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "underlying_icd_norm"},
        "axes": {"diagnostic": "ICD10"},
        "provenance": ["fixture"],
        "warnings": [],
        "lineage_hash": "abc",
        "registry_hash": "registry",
        "materialization_state": "metadata_only",
    }


def test_slice15b_selects_only_promotable_metadata_only_fields() -> None:
    plan = {
        "promotions": [
            {"status": "planned", "action": "promote", "field": _field()},
            {"status": "blocked", "blocked": True, "field": {**_field(), "field_id": "blocked"}},
            {"status": "planned", "action": "defer", "field": {**_field(), "field_id": "deferred"}},
        ]
    }
    promoted, skipped, blocked = planned_promotion_fields(plan)
    assert [field["field_id"] for field in promoted] == ["efg_substrate__SIM_DO__underlying_icd_norm"]
    assert [item["field_id"] for item in skipped] == ["deferred"]
    assert [item["field_id"] for item in blocked] == ["blocked"]


def test_slice15b_builds_output_rows_for_validator_contract() -> None:
    field = _field()
    v = v_field_row(field)
    q = q_tensor_row(field)
    vd = variable_dictionary_row(field)
    assert v["field_id"] == field["field_id"]
    assert v["unit"] == "ICD10"
    assert v["state"] == "quarantined_descriptive"
    assert v["materialization_state"] == "metadata_only"
    assert q["field_id"] == field["field_id"]
    assert q["state"] == "quarantined_descriptive"
    assert vd["field_id"] == field["field_id"]
    assert vd["estimand_label"] == "metadata_only_observer_not_model_estimand"
'''
    integration = r'''
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def test_slice15b_applies_promotion_plan_to_17_key_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    field = {
        "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
        "name": "SIM-DO underlying ICD observer",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "ICD10",
        "aggregation": "non_aggregable",
        "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "underlying_icd_norm"},
        "axes": {"diagnostic": "ICD10"},
        "provenance": ["fixture"],
        "warnings": [],
        "lineage_hash": "abc",
        "registry_hash": "registry",
        "materialization_state": "metadata_only",
    }
    plan_path = run_dir / "Tables" / "efg_promotion_plan.json"
    _write_json(plan_path, {"schema_version": "1.0", "promotions": [{"status": "planned", "action": "promote", "field": field}]})

    result = apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
    assert result.promoted_field_count == 1
    assert result.output_validation_ok is True

    v_ids = {row["field_id"] for row in pq.read_table(run_dir / "V_fields.parquet").to_pylist()}
    q_ids = {row["field_id"] for row in pq.read_table(run_dir / "Q_tensor.parquet").to_pylist()}
    vd_ids = {row["field_id"] for row in pq.read_table(run_dir / "VariableDictionary.parquet").to_pylist()}
    assert field["field_id"] in v_ids
    assert field["field_id"] in q_ids
    assert field["field_id"] in vd_ids

    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["efg_promotion_apply_gate"]["promoted_field_count"] == 1
    assert validate_output_bundle(run_dir=str(run_dir)).ok


def test_slice15b_apply_is_idempotent_by_field_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    plan_path = run_dir / "Tables" / "efg_promotion_plan.json"
    field = {
        "field_id": "efg_substrate__SIM_DO__year",
        "name": "SIM-DO year observer",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "year",
        "aggregation": "non_aggregable",
        "role": ["source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "year"},
        "axes": {"time": "year"},
        "provenance": ["fixture"],
        "materialization_state": "metadata_only",
    }
    _write_json(plan_path, {"promotions": [{"status": "planned", "action": "promote", "field": field}]})
    apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
    apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
    rows = pq.read_table(run_dir / "V_fields.parquet").to_pylist()
    assert sum(1 for row in rows if row["field_id"] == field["field_id"]) == 1
'''
    write("tests/unit/test_slice15b_efg_promotion_apply.py", unit)
    write("tests/integration/test_slice15b_apply_efg_promotions_to_bundle.py", integration)


def write_audit() -> None:
    code = r'''
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must still contain exactly one _run_compile_impl")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        before_keys = sorted(OUTPUT_BUNDLE_FILES.keys())
        field = {
            "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
            "name": "SIM-DO underlying ICD observer",
            "kind": "observer_proxy",
            "carrier": "Deaths",
            "unit": "ICD10",
            "aggregation": "non_aggregable",
            "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
            "source": ["SIM-DO"],
            "support": {"column": "underlying_icd_norm"},
            "axes": {"diagnostic": "ICD10"},
            "provenance": ["fixture"],
            "warnings": [],
            "lineage_hash": "abc",
            "registry_hash": "registry",
            "materialization_state": "metadata_only",
        }
        plan_path = run_dir / "Tables" / "efg_promotion_plan.json"
        plan_path.write_text(json.dumps({"promotions": [{"status": "planned", "action": "promote", "field": field}]}), encoding="utf-8")
        result = apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
        if result.promoted_field_count != 1:
            errors.append("expected exactly one promoted field")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.extend(validation.errors)
        v_ids = {row["field_id"] for row in pq.read_table(run_dir / "V_fields.parquet").to_pylist()}
        q_ids = {row["field_id"] for row in pq.read_table(run_dir / "Q_tensor.parquet").to_pylist()}
        vd_ids = {row["field_id"] for row in pq.read_table(run_dir / "VariableDictionary.parquet").to_pylist()}
        if field["field_id"] not in v_ids or field["field_id"] not in q_ids or field["field_id"] not in vd_ids:
            errors.append("promoted field missing from V_fields/Q_tensor/VariableDictionary")
        after_keys = sorted(OUTPUT_BUNDLE_FILES.keys())
        if before_keys != after_keys:
            errors.append("output bundle first-class key registry changed")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 15B EFG promotion apply boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write("scripts/dev/audits/audit_slice15b_efg_promotion_apply.py", code)


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    sys.path.insert(0, str(ROOT / "src"))
    from pegasus.efg.promotion_apply import planned_promotion_fields, v_field_row
    field = {"field_id": "f", "materialization_state": "metadata_only"}
    promoted, skipped, blocked = planned_promotion_fields({"promotions": [{"status": "planned", "action": "promote", "field": field}]})
    if len(promoted) != 1 or skipped or blocked:
        fail("self-validate promotion selection failed")
    if v_field_row(field)["field_id"] != "f":
        fail("self-validate V_fields row failed")


def main() -> None:
    preflight()
    write_promotion_apply_module()
    write_workflow_module()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 15B updater applied: EFG promotion apply boundary.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
