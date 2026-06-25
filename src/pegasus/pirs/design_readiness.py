
"""PIRS design-readiness gate for planned design contracts.

Slice 16D intentionally does not build a numerical design matrix.  It reads a
Slice 16C PIRS design plan and decides whether the selected fields have enough
tensor-backed information for a later matrix builder to run safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

try:  # pragma: no cover - pyarrow may be optional in static tooling, but is present in tests.
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None  # type: ignore[assignment]


BLOCKING_FIELD_STATES: frozenset[str] = frozenset({
    "blocked",
    "illegal_excluded",
    "quarantined_descriptive",
})
BLOCKING_MATERIALIZATION_STATES: frozenset[str] = frozenset({
    "metadata_only",
    "not_materialized",
    "blocked",
})
PLACEHOLDER_WARNING_TOKENS: frozenset[str] = frozenset({
    "q_tensor_unmaterialized_candidate_no_numerical_tensor",
    "efg_promotion_metadata_only",
})


@dataclass(frozen=True)
class DesignReadinessDecision:
    field_id: str
    role: str
    ready: bool
    reasons: tuple[str, ...]
    field: dict[str, Any]
    q_state: dict[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "role": self.role,
            "ready": self.ready,
            "reasons": list(self.reasons),
            "field": self.field,
            "q_state": self.q_state,
        }


@dataclass(frozen=True)
class DesignReadinessResult:
    status: str
    design_matrix_state: str
    model_fit_state: str
    residual_state: str
    hsic_state: str
    ready_field_count: int
    blocked_field_count: int
    decisions: tuple[DesignReadinessDecision, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "slice": "16D",
            "status": self.status,
            "ready": self.ready,
            "design_matrix_state": self.design_matrix_state,
            "model_fit_state": self.model_fit_state,
            "residual_state": self.residual_state,
            "hsic_state": self.hsic_state,
            "ready_field_count": self.ready_field_count,
            "blocked_field_count": self.blocked_field_count,
            "decisions": [decision.as_manifest() for decision in self.decisions],
            "warnings": list(self.warnings),
        }


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return p


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or pq is None:
        return []
    table = pq.read_table(path)
    return table.to_pylist()


def _json_cell(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _as_list(value: Any) -> list[Any]:
    value = _json_cell(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _field_id(value: dict[str, Any]) -> str | None:
    raw = value.get("field_id") or value.get("id") or value.get("name")
    return str(raw) if raw not in (None, "") else None


def _field_rows_by_id(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(run_dir)
    rows = _read_parquet_rows(root / "V_fields.parquet")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        fid = _field_id(row)
        if fid is not None:
            out[fid] = {key: _json_cell(value) for key, value in row.items()}
    return out


def _q_rows_by_id(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(run_dir)
    rows = _read_parquet_rows(root / "Q_tensor.parquet")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        fid = _field_id(row)
        if fid is not None:
            out[fid] = {key: _json_cell(value) for key, value in row.items()}
    return out


def _selected_field_ids(plan: dict[str, Any]) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for key, role in (
        ("selected_outcome_field_id", "outcome"),
        ("outcome_field_id", "outcome"),
        ("selected_offset_field_id", "offset"),
        ("offset_field_id", "offset"),
    ):
        value = plan.get(key)
        if value not in (None, ""):
            selected.append((str(value), role))
    covariates = plan.get("selected_covariate_field_ids")
    if covariates is None:
        covariates = plan.get("covariate_field_ids")
    for value in _as_list(covariates):
        if value not in (None, ""):
            selected.append((str(value), "covariate"))
    nested_selection = plan.get("selection")
    if isinstance(nested_selection, dict):
        for key, role in (("outcome", "outcome"), ("offset", "offset")):
            value = nested_selection.get(key)
            if isinstance(value, dict):
                value = value.get("field_id")
            if value not in (None, ""):
                selected.append((str(value), role))
        for value in _as_list(nested_selection.get("covariates")):
            if isinstance(value, dict):
                value = value.get("field_id")
            if value not in (None, ""):
                selected.append((str(value), "covariate"))
    # Preserve order while de-duplicating field/role pairs.
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in selected:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _state(row: dict[str, Any]) -> str:
    value = row.get("state") or row.get("field_state") or row.get("q_state")
    return str(value) if value not in (None, "") else "unknown"


def _materialization_state(row: dict[str, Any]) -> str:
    value = row.get("materialization_state") or row.get("materialized_state")
    return str(value) if value not in (None, "") else "unknown"


def _warnings(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("warnings", "warning_codes", "warning"):
        for value in _as_list(row.get(key)):
            if value not in (None, ""):
                values.append(str(value))
    return list(dict.fromkeys(values))


def design_readiness_rejection_reasons(*, field: dict[str, Any] | None, q_state: dict[str, Any] | None) -> tuple[str, ...]:
    reasons: list[str] = []
    if not field:
        reasons.append("field_missing_from_V_fields")
        return tuple(reasons)
    if not q_state:
        reasons.append("q_tensor_row_missing")
    field_state = _state(field)
    if field_state in BLOCKING_FIELD_STATES:
        reasons.append(f"field_state_not_model_ready:{field_state}")
    materialization_state = _materialization_state(field)
    if materialization_state in BLOCKING_MATERIALIZATION_STATES:
        reasons.append(f"materialization_state_not_tensor_backed:{materialization_state}")
    dashboard_safe = _as_bool(field.get("dashboard_safe"))
    if dashboard_safe is False:
        reasons.append("field_not_dashboard_safe")
    if q_state:
        q_status = _state(q_state)
        if q_status in BLOCKING_FIELD_STATES:
            reasons.append(f"q_state_not_model_ready:{q_status}")
        for warning in _warnings(q_state):
            if warning in PLACEHOLDER_WARNING_TOKENS:
                reasons.append(f"q_tensor_unmaterialized_candidate:{warning}")
        n_eff = q_state.get("n_eff")
        try:
            if n_eff is not None and float(n_eff) <= 0:
                reasons.append("q_tensor_n_eff_nonpositive")
        except (TypeError, ValueError):
            reasons.append("q_tensor_n_eff_not_numeric")
        missingness = q_state.get("missingness")
        try:
            if missingness is not None and float(missingness) >= 1.0:
                reasons.append("q_tensor_missingness_complete")
        except (TypeError, ValueError):
            reasons.append("q_tensor_missingness_not_numeric")
    return tuple(dict.fromkeys(reasons))


def build_pirs_design_readiness_gate(*, run_dir: str | Path, design_plan: str | Path | None = None) -> DesignReadinessResult:
    root = Path(run_dir)
    plan_path = Path(design_plan) if design_plan is not None else root / "Tables" / "pirs_design_plan.json"
    plan = _load_json(plan_path)
    fields = _field_rows_by_id(root)
    q_rows = _q_rows_by_id(root)
    selected = _selected_field_ids(plan)
    warnings: list[str] = []
    decisions: list[DesignReadinessDecision] = []
    if not selected:
        warnings.append("pirs_design_plan_has_no_selected_fields")
    for field_id, role in selected:
        field = fields.get(field_id, {})
        q_state = q_rows.get(field_id, {})
        reasons = design_readiness_rejection_reasons(field=field or None, q_state=q_state or None)
        decisions.append(
            DesignReadinessDecision(
                field_id=field_id,
                role=role,
                ready=not reasons,
                reasons=reasons,
                field=field,
                q_state=q_state,
            )
        )
    blocked = sum(1 for decision in decisions if not decision.ready)
    ready = sum(1 for decision in decisions if decision.ready)
    if not decisions:
        status = "blocked"
        warnings.append("pirs_design_readiness_no_fields_to_evaluate")
    elif blocked:
        status = "blocked"
    else:
        status = "ready"
    return DesignReadinessResult(
        status=status,
        design_matrix_state="ready_to_build" if status == "ready" else "blocked_until_tensor_backed_fields",
        model_fit_state="not_started",
        residual_state="not_started",
        hsic_state="not_started",
        ready_field_count=ready,
        blocked_field_count=blocked,
        decisions=tuple(decisions),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def design_readiness_summary(result: DesignReadinessResult, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "status": result.status,
        "ready": result.ready,
        "design_matrix_state": result.design_matrix_state,
        "model_fit_state": result.model_fit_state,
        "residual_state": result.residual_state,
        "hsic_state": result.hsic_state,
        "ready_field_count": result.ready_field_count,
        "blocked_field_count": result.blocked_field_count,
        "decision_count": len(result.decisions),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "warnings": list(result.warnings),
    }


def write_pirs_design_readiness_manifest(
    *,
    run_dir: str | Path,
    design_plan: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    out = Path(output) if output is not None else root / "Tables" / "pirs_design_readiness.json"
    result = build_pirs_design_readiness_gate(run_dir=root, design_plan=design_plan)
    payload = result.as_manifest()
    payload["source_design_plan"] = str(design_plan) if design_plan is not None else str(root / "Tables" / "pirs_design_plan.json")
    _write_json(out, payload)
    return {"manifest_path": str(out), "pirs_design_readiness_gate": design_readiness_summary(result, manifest_path=out)}


def _merge_json_gate(path: Path, gate_name: str, summary: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except json.JSONDecodeError:
            payload = {}
    payload[gate_name] = summary
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def attach_pirs_design_readiness_to_run(
    *,
    run_dir: str | Path,
    design_plan: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    result = write_pirs_design_readiness_manifest(run_dir=root, design_plan=design_plan, output=output)
    summary = dict(result["pirs_design_readiness_gate"])
    for filename in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        _merge_json_gate(root / filename, "pirs_design_readiness_gate", summary)
    return result
