
from __future__ import annotations

"""Non-mutating PIRS design-plan boundary.

Slice 16C consumes a Slice 16B PIRS selection plan and emits an auditable
planned-only design contract.  It intentionally does not read observation
matrices, fit models, create residuals, or write first-class output-bundle keys.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pegasus.pirs.spatial import select_spatial_effect_mode


DESIGN_PLAN_SCHEMA_VERSION = "1.0"
DESIGN_PLAN_ARTIFACT = "pirs_design_plan"
DESIGN_GATE_KEY = "pirs_design_gate"
DEFAULT_SELECTION_PLAN = Path("Tables/pirs_selection_plan.json")
DEFAULT_DESIGN_PLAN = Path("Tables/pirs_design_plan.json")
JSON_ATTACH_TARGETS: tuple[str, ...] = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)


class PIRSDesignPlanError(ValueError):
    """Raised when a PIRS selection plan cannot be converted to a design plan."""


@dataclass(frozen=True)
class PIRSDesignTerm:
    term_id: str
    role: str
    field_id: str | None
    transform: str
    dtype: str
    required: bool
    source: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "role": self.role,
            "field_id": self.field_id,
            "transform": self.transform,
            "dtype": self.dtype,
            "required": self.required,
            "source": self.source,
        }


@dataclass(frozen=True)
class PIRSDesignPlan:
    status: str
    design_matrix_state: str
    budget: str
    residual_mode: str
    family: str
    spatial_effect_mode: str
    spatial_effect: dict[str, Any]
    fold_scheme: dict[str, Any]
    outcome_field_id: str | None
    covariate_field_ids: tuple[str, ...]
    offset_field_id: str | None
    terms: tuple[PIRSDesignTerm, ...]
    rejected: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]
    source_selection_plan_hash: str | None = None

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def covariate_count(self) -> int:
        return len(self.covariate_field_ids)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DESIGN_PLAN_SCHEMA_VERSION,
            "artifact": DESIGN_PLAN_ARTIFACT,
            "status": self.status,
            "design_matrix_state": self.design_matrix_state,
            "budget": self.budget,
            "residual_mode": self.residual_mode,
            "family": self.family,
            "spatial_effect_mode": self.spatial_effect_mode,
            "spatial_effect": dict(self.spatial_effect),
            "fold_scheme": self.fold_scheme,
            "outcome_field_id": self.outcome_field_id,
            "covariate_field_ids": list(self.covariate_field_ids),
            "offset_field_id": self.offset_field_id,
            "terms": [term.as_manifest() for term in self.terms],
            "rejected": list(self.rejected),
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
            "source_selection_plan_hash": self.source_selection_plan_hash,
            "non_mutating": True,
            "model_fit_state": "not_started",
            "residual_state": "not_started",
            "hsic_state": "not_started",
            "created_at": _utc_now(),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return p


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            item = item.get("field_id") or item.get("id") or item.get("name")
        text = _string_or_none(item)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _selection_payload(selection_plan: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(selection_plan, (str, Path)):
        return _load_json(selection_plan)
    if isinstance(selection_plan, Mapping):
        return dict(selection_plan)
    raise PIRSDesignPlanError(f"Unsupported selection plan type: {type(selection_plan)!r}")


def _selected_outcome(plan: Mapping[str, Any]) -> str | None:
    direct = _string_or_none(plan.get("selected_outcome_field_id"))
    if direct:
        return direct
    selection = _as_dict(plan.get("selection"))
    direct = _string_or_none(selection.get("selected_outcome_field_id") or selection.get("outcome_field_id"))
    if direct:
        return direct
    outcome = plan.get("selected_outcome") or selection.get("outcome")
    if isinstance(outcome, Mapping):
        return _string_or_none(outcome.get("field_id") or outcome.get("id"))
    return _string_or_none(outcome)


def _selected_covariates(plan: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("selected_covariate_field_ids", "covariate_field_ids", "selected_covariates"):
        values = _string_list(plan.get(key))
        if values:
            return values
    selection = _as_dict(plan.get("selection"))
    for key in ("selected_covariate_field_ids", "covariate_field_ids", "covariates"):
        values = _string_list(selection.get(key))
        if values:
            return values
    return ()


def _selected_offset(plan: Mapping[str, Any]) -> str | None:
    direct = _string_or_none(plan.get("selected_offset_field_id") or plan.get("offset_field_id"))
    if direct:
        return direct
    selection = _as_dict(plan.get("selection"))
    direct = _string_or_none(selection.get("selected_offset_field_id") or selection.get("offset_field_id"))
    if direct:
        return direct
    offset = plan.get("selected_offset") or selection.get("offset")
    if isinstance(offset, Mapping):
        return _string_or_none(offset.get("field_id") or offset.get("id"))
    return _string_or_none(offset)


def _rejected(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rejected: list[dict[str, Any]] = []
    for key in ("gate_rejected", "selection_rejected", "rejected", "warnings_as_rejections"):
        for item in _as_list(plan.get(key)):
            if isinstance(item, Mapping):
                rejected.append(dict(item))
            elif item is not None:
                rejected.append({"reason": str(item)})
    return tuple(rejected)


def _family(plan: Mapping[str, Any], offset_field_id: str | None) -> str:
    direct = _string_or_none(plan.get("family"))
    if direct:
        return direct
    diagnostics = _as_dict(plan.get("diagnostics"))
    direct = _string_or_none(diagnostics.get("family"))
    if direct:
        return direct
    return "poisson_rate" if offset_field_id else "gaussian_identity"


def _fold_scheme(plan: Mapping[str, Any]) -> dict[str, Any]:
    fold = _as_dict(plan.get("fold_scheme"))
    if fold:
        return fold
    diagnostics = _as_dict(plan.get("diagnostics"))
    fold = _as_dict(diagnostics.get("fold_scheme"))
    if fold:
        return fold
    return {"mode": "not_declared", "fold_count": None, "source": "selection_plan_missing_fold_scheme"}


def _spatial_effect_payload(plan: Mapping[str, Any], *, budget: str) -> tuple[str, tuple[str, ...], dict[str, Any]]:
    direct = _string_or_none(plan.get("spatial_effect_mode"))
    if direct:
        return direct, (), {"mode": direct, "reason": "declared_by_selection_plan", "warnings": []}
    diagnostics = _as_dict(plan.get("diagnostics"))
    direct = _string_or_none(diagnostics.get("spatial_effect_mode"))
    if direct:
        return direct, (), {"mode": direct, "reason": "declared_by_selection_diagnostics", "warnings": []}

    def first_present(key: str) -> Any:
        return plan[key] if key in plan and plan[key] is not None else diagnostics.get(key)

    selector = select_spatial_effect_mode(
        budget=budget,
        time_period_count=first_present("time_period_count"),
        spatial_missingness=first_present("spatial_missingness"),
        moran_i=first_present("moran_i"),
    )
    manifest = selector.as_manifest()
    adjacency_path = first_present("adjacency_path") or first_present("geo_adjacency_path")
    if adjacency_path not in (None, ""):
        manifest["adjacency_path"] = str(adjacency_path)
    return selector.mode, selector.warnings, manifest


def _selection_hash(plan: Mapping[str, Any]) -> str | None:
    for key in ("selection_plan_hash", "source_selection_plan_hash", "manifest_hash", "registry_hash"):
        value = _string_or_none(plan.get(key))
        if value:
            return value
    return None


def _term(term_id: str, role: str, field_id: str | None, transform: str, *, required: bool, source: str) -> PIRSDesignTerm:
    return PIRSDesignTerm(
        term_id=term_id,
        role=role,
        field_id=field_id,
        transform=transform,
        dtype="float64",
        required=required,
        source=source,
    )


def build_pirs_design_plan(
    selection_plan: str | Path | Mapping[str, Any],
    *,
    budget: str | None = None,
) -> PIRSDesignPlan:
    """Build a non-mutating design plan from a PIRS selection plan."""

    payload = _selection_payload(selection_plan)
    plan_budget = str(budget or payload.get("budget") or "fast")
    outcome = _selected_outcome(payload)
    covariates = _selected_covariates(payload)
    offset = _selected_offset(payload)
    rejected = _rejected(payload)
    warnings = [str(w) for w in _as_list(payload.get("warnings")) if w]
    if not outcome:
        warnings.append("pirs_design_plan_missing_outcome")
    if not covariates:
        warnings.append("pirs_design_plan_missing_covariates")
    spatial_mode, spatial_warnings, spatial_manifest = _spatial_effect_payload(payload, budget=plan_budget)
    warnings.extend(spatial_warnings)

    terms: list[PIRSDesignTerm] = [_term("intercept", "intercept", None, "constant_one", required=True, source="design_plan")]
    if outcome:
        terms.append(_term("response", "outcome", outcome, "identity", required=True, source="selection_plan"))
    for i, field_id in enumerate(covariates, start=1):
        terms.append(_term(f"covariate_{i:03d}", "covariate", field_id, "identity", required=True, source="selection_plan"))
    if offset:
        terms.append(_term("offset", "offset", offset, "log_exposure_offset", required=False, source="selection_plan"))

    status = "planned" if outcome and covariates else "blocked"
    return PIRSDesignPlan(
        status=status,
        design_matrix_state="planned_only",
        budget=plan_budget,
        residual_mode=str(payload.get("residual_mode") or "in_sample"),
        family=_family(payload, offset),
        spatial_effect_mode=spatial_mode,
        spatial_effect=spatial_manifest,
        fold_scheme=_fold_scheme(payload),
        outcome_field_id=outcome,
        covariate_field_ids=covariates,
        offset_field_id=offset,
        terms=tuple(terms),
        rejected=rejected,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=("pirs_selection_plan", "slice16c_design_plan_boundary"),
        source_selection_plan_hash=_selection_hash(payload),
    )


def pirs_design_plan_summary(plan: PIRSDesignPlan | Mapping[str, Any], *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    payload = plan.as_manifest() if isinstance(plan, PIRSDesignPlan) else dict(plan)
    covariates = _string_list(payload.get("covariate_field_ids"))
    warnings = _as_list(payload.get("warnings"))
    rejected = _as_list(payload.get("rejected"))
    return {
        "status": payload.get("status"),
        "design_matrix_state": payload.get("design_matrix_state"),
        "budget": payload.get("budget"),
        "residual_mode": payload.get("residual_mode"),
        "family": payload.get("family"),
        "fold_scheme": payload.get("fold_scheme"),
        "outcome_field_id": payload.get("outcome_field_id"),
        "covariate_count": len(covariates),
        "term_count": len(_as_list(payload.get("terms"))),
        "rejected_count": len(rejected),
        "warnings_count": len(warnings),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "non_mutating": True,
        "model_fit_state": payload.get("model_fit_state", "not_started"),
        "residual_state": payload.get("residual_state", "not_started"),
        "hsic_state": payload.get("hsic_state", "not_started"),
    }


def write_pirs_design_plan(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    selection_plan = selection_plan if selection_plan is not None else run_path / DEFAULT_SELECTION_PLAN
    output_path = Path(output) if output is not None else run_path / DEFAULT_DESIGN_PLAN
    plan = build_pirs_design_plan(selection_plan, budget=budget)
    payload = plan.as_manifest()
    payload["summary"] = pirs_design_plan_summary(plan, manifest_path=output_path)
    _write_json(output_path, payload)
    return payload


def _attach_summary(run_dir: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(run_dir)
    for rel in JSON_ATTACH_TARGETS:
        path = root / rel
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[DESIGN_GATE_KEY] = dict(summary)
        _write_json(path, payload)


def attach_pirs_design_plan_to_run(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    payload = write_pirs_design_plan(
        run_dir=run_dir,
        selection_plan=selection_plan,
        output=output,
        budget=budget,
    )
    summary = pirs_design_plan_summary(payload, manifest_path=Path(output) if output is not None else Path(run_dir) / DEFAULT_DESIGN_PLAN)
    _attach_summary(run_dir, summary)
    payload["summary"] = summary
    return payload
