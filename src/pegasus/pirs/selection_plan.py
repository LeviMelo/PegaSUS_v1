
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pegasus.efg.empirical_compression import empirical_compress
from pegasus.pirs.crossfit import assert_standard_deep_not_in_sample, fold_scheme_for_budget
from pegasus.pirs.design_matrix import _q_value_vector, _read_rows as _read_q_rows
from pegasus.pirs.diagnostics import build_pirs_diagnostics
from pegasus.pirs.families import exposure_offset_source, family_for_outcome
from pegasus.pirs.field_selection import select_fields_for_pirs
from pegasus.pirs.schemas import FieldCandidate, PIRSSelectionResult

Budget = Literal["fast", "standard", "deep"]


class PIRSSelectionPlanError(ValueError):
    """Raised when a PIRS selection plan cannot be built from a candidate gate."""


@dataclass(frozen=True)
class PIRSSelectionPlan:
    schema_version: str
    gate: str
    status: str
    budget: Budget
    top_k: int
    residual_mode: str
    fold_scheme: dict[str, Any]
    family: str | None
    exposure_offset_source: str | None
    selected_outcome: FieldCandidate | None
    selected_covariates: tuple[FieldCandidate, ...]
    selected_offset: FieldCandidate | None
    selection_rejected: tuple[dict[str, Any], ...]
    gate_rejected: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    candidate_manifest_path: str
    warnings: tuple[str, ...]

    @property
    def selected_outcome_field_id(self) -> str | None:
        return self.selected_outcome.field_id if self.selected_outcome else None

    @property
    def selected_covariate_field_ids(self) -> tuple[str, ...]:
        return tuple(candidate.field_id for candidate in self.selected_covariates)

    @property
    def selected_offset_field_id(self) -> str | None:
        return self.selected_offset.field_id if self.selected_offset else None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate": self.gate,
            "status": self.status,
            "budget": self.budget,
            "top_k": self.top_k,
            "residual_mode": self.residual_mode,
            "fold_scheme": self.fold_scheme,
            "family": self.family,
            "exposure_offset_source": self.exposure_offset_source,
            "selected_outcome_field_id": self.selected_outcome_field_id,
            "selected_covariate_field_ids": list(self.selected_covariate_field_ids),
            "selected_offset_field_id": self.selected_offset_field_id,
            "selected_outcome": asdict(self.selected_outcome) if self.selected_outcome else None,
            "selected_covariates": [asdict(candidate) for candidate in self.selected_covariates],
            "selected_offset": asdict(self.selected_offset) if self.selected_offset else None,
            "selection_rejected": list(self.selection_rejected),
            "gate_rejected": list(self.gate_rejected),
            "diagnostics": dict(self.diagnostics),
            "candidate_manifest_path": self.candidate_manifest_path,
            "warnings": list(self.warnings),
            "non_behavior": {
                "model_fitted": False,
                "design_matrix_materialized": False,
                "residuals_materialized": False,
                "hsic_triggered": False,
            },
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PIRSSelectionPlanError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload), encoding="utf-8")


def _tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _candidate_from_mapping(payload: dict[str, Any]) -> FieldCandidate:
    return FieldCandidate(
        field_id=str(payload["field_id"]),
        role=str(payload.get("role", "covariate")),  # type: ignore[arg-type]
        utility=float(payload.get("utility", 0.0) or 0.0),
        q_state=str(payload.get("q_state", "verified")),  # type: ignore[arg-type]
        carrier=str(payload.get("carrier", "unknown")),
        unit=str(payload.get("unit", "unknown")),
        support=dict(payload.get("support") or {}),
        variance=None if payload.get("variance") is None else float(payload.get("variance")),
        warnings=_tuple_str(payload.get("warnings")),
        provenance=_tuple_str(payload.get("provenance")),
    )


def load_pirs_candidate_manifest(candidate_manifest: str | Path) -> tuple[list[FieldCandidate], tuple[dict[str, Any], ...], dict[str, Any]]:
    path = Path(candidate_manifest)
    payload = _load_json(path)
    if payload.get("gate") != "pirs_candidate_gate":
        raise PIRSSelectionPlanError(f"Not a PIRS candidate gate manifest: {path}")
    candidates = [_candidate_from_mapping(item) for item in payload.get("candidates", [])]
    rejected = tuple(dict(item) for item in payload.get("rejected", []))
    return candidates, rejected, payload


def _support_kind(candidates: list[FieldCandidate]) -> str:
    for candidate in candidates:
        support = candidate.support or {}
        if support.get("months") or support.get("month"):
            return "monthly_municipal_panel"
    return "annual_municipal_panel"


def build_pirs_selection_plan(
    *,
    candidate_manifest: str | Path,
    budget: Budget = "fast",
) -> PIRSSelectionPlan:
    candidate_manifest_path = Path(candidate_manifest)
    candidates, gate_rejected, raw_manifest = load_pirs_candidate_manifest(candidate_manifest_path)
    selection = select_fields_for_pirs(candidates, budget=budget)
    support_kind = _support_kind(candidates)
    fold = fold_scheme_for_budget(budget=budget, support_kind=support_kind)
    assert_standard_deep_not_in_sample(budget, fold.residual_mode)

    family = None
    offset_source = None
    warnings: list[str] = []
    if selection.selected_outcome is None:
        status = "blocked_no_outcome"
        warnings.append("pirs_selection_has_no_model_eligible_outcome")
    else:
        status = "planned"
        family = family_for_outcome(outcome=selection.selected_outcome, offset=selection.selected_offset)
        try:
            offset_source = exposure_offset_source(family=family, offset=selection.selected_offset)
        except ValueError as exc:
            status = "blocked_invalid_offset"
            warnings.append(str(exc))
            offset_source = None
    diagnostics = build_pirs_diagnostics(
        selection=selection,
        fold_scheme=fold,
        exposure_offset_source=offset_source,
    ).as_manifest()
    if raw_manifest.get("rejected_count", 0):
        warnings.append("pirs_candidate_gate_rejected_fields_before_selection")
    if fold.residual_mode == "in_sample":
        warnings.append("fast_budget_in_sample_residuals_not_for_standard_hsic")
    return PIRSSelectionPlan(
        schema_version="1.0",
        gate="pirs_selection_plan",
        status=status,
        budget=budget,
        top_k=selection.top_k,
        residual_mode=fold.residual_mode,
        fold_scheme=fold.as_manifest(),
        family=family,
        exposure_offset_source=offset_source,
        selected_outcome=selection.selected_outcome,
        selected_covariates=selection.selected_covariates,
        selected_offset=selection.selected_offset,
        selection_rejected=selection.rejected,
        gate_rejected=gate_rejected,
        diagnostics=diagnostics,
        candidate_manifest_path=str(candidate_manifest_path),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def selection_plan_summary(plan: PIRSSelectionPlan, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "gate": "pirs_selection_plan",
        "status": plan.status,
        "budget": plan.budget,
        "residual_mode": plan.residual_mode,
        "candidate_manifest_path": plan.candidate_manifest_path,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "selected_outcome_field_id": plan.selected_outcome_field_id,
        "selected_covariate_count": len(plan.selected_covariates),
        "selected_offset_field_id": plan.selected_offset_field_id,
        "gate_rejected_count": len(plan.gate_rejected),
        "selection_rejected_count": len(plan.selection_rejected),
        "model_fitted": False,
        "design_matrix_materialized": False,
        "residuals_materialized": False,
        "hsic_triggered": False,
    }


def _empirical_compression_for_plan(run_dir: Path, plan: PIRSSelectionPlan) -> Any | None:
    """MSD §3.15.2 Stage-2: fold empirically-equivalent (C_emp >= 0.98) selected
    covariates into higher-utility canonicals, on materialized Q_tensor vectors.

    Runs only post-TopK (here) and only when value vectors exist; if Q_tensor is
    not yet materialized every vector is missing and the report is a no-op.
    """
    covariates = plan.selected_covariates
    if len(covariates) < 2:
        return None
    q_rows = _read_q_rows(run_dir / "Q_tensor.parquet")
    q_by_id = {str(row.get("field_id")): row for row in q_rows if row.get("field_id") not in (None, "")}
    items: list[tuple[str, float, list[float] | None]] = []
    for candidate in covariates:
        q = q_by_id.get(candidate.field_id)
        vector = _q_value_vector(q) if q is not None else None
        items.append((candidate.field_id, float(candidate.utility), vector))
    return empirical_compress(items)


def write_pirs_selection_plan(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    root = Path(run_dir)
    candidate_path = Path(candidate_manifest) if candidate_manifest is not None else root / "Tables" / "pirs_field_candidates.json"
    output_path = Path(output) if output is not None else root / "Tables" / "pirs_selection_plan.json"
    plan = build_pirs_selection_plan(candidate_manifest=candidate_path, budget=budget)
    manifest = plan.as_manifest()
    manifest["manifest_path"] = str(output_path)
    manifest["summary"] = selection_plan_summary(plan, manifest_path=output_path)

    # Stage-2 empirical compression realises (not just reports) by filtering the
    # selected covariate set the design plan consumes downstream. Suppressed
    # nodes are retained in V_fields/Q_tensor and recorded here — no data lost.
    compression = _empirical_compression_for_plan(root, plan)
    if compression is not None and compression.suppressed_count:
        survivors = [
            field_id
            for field_id in plan.selected_covariate_field_ids
            if compression.canonical_by_field_id.get(field_id, field_id) == field_id
        ]
        manifest["empirical_compression"] = compression.as_manifest()
        manifest["selected_covariate_field_ids_precompression"] = list(plan.selected_covariate_field_ids)
        manifest["selected_covariate_field_ids"] = survivors
        survivor_set = set(survivors)
        manifest["selected_covariates"] = [
            covariate
            for covariate in manifest["selected_covariates"]
            if covariate.get("field_id") in survivor_set
        ]
        manifest["summary"]["selected_covariate_count"] = len(survivors)
        manifest["summary"]["empirical_compression_suppressed"] = compression.suppressed_count
    elif compression is not None:
        manifest["empirical_compression"] = compression.as_manifest()

    _write_json(output_path, manifest)
    return manifest


def attach_pirs_selection_plan_to_run(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = write_pirs_selection_plan(run_dir=root, candidate_manifest=candidate_manifest, output=output, budget=budget)
    summary = dict(manifest["summary"])
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / rel
        if not path.exists():
            continue
        payload = _load_json(path)
        payload["pirs_selection_gate"] = summary
        _write_json(path, payload)
    return summary
