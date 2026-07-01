"""Intent-derived compiler stage plans and proof-carrying skip contracts.

Macro-Slice 26A/26B changes the meaning of optional compiler stages from
free-form telemetry labels into an auditable contract.  A stage may be marked
``skipped`` only when the current intent did not request it or when a typed
source/artifact gate explicitly proves that skipping is legal.  This module is
pure metadata logic; it does not run population, ST-DFM, PIRS, HSIC, or geo
kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

OPTIONAL_STAGE_IDS: tuple[str, ...] = (
    "geo_support",
    "population_solver",
    "stdfm",
    "pirs_model",
    "pirs_hsic",
)

EXECUTED_STATUS = {"success"}
NON_EXECUTED_STATUS = {"skipped", "blocked", "failed"}


@dataclass(frozen=True)
class CompilerStageRequirement:
    stage_id: str
    requested: bool
    request_source: str
    required_for_level3: bool
    can_execute: bool
    skip_allowed: bool
    skip_reason: str | None = None
    blocked_reason: str | None = None
    executor: str | None = None
    expected_artifacts: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "requested": self.requested,
            "request_source": self.request_source,
            "required_for_level3": self.required_for_level3,
            "can_execute": self.can_execute,
            "skip_allowed": self.skip_allowed,
            "skip_reason": self.skip_reason,
            "blocked_reason": self.blocked_reason,
            "executor": self.executor,
            "expected_artifacts": list(self.expected_artifacts),
        }


@dataclass(frozen=True)
class CompilerStagePlan:
    schema_version: str
    plan_id: str
    intent_budget: str | None
    intent_geo_mode: str | None
    intent_population_mode: str | None
    source_manifest_supplied: bool
    require_materialized_external: bool
    stages: tuple[CompilerStageRequirement, ...]

    def by_stage(self) -> dict[str, CompilerStageRequirement]:
        return {stage.stage_id: stage for stage in self.stages}

    def skip_reason_map(self) -> dict[str, str]:
        return {stage.stage_id: stage.skip_reason or "" for stage in self.stages if stage.skip_allowed}

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "intent_budget": self.intent_budget,
            "intent_geo_mode": self.intent_geo_mode,
            "intent_population_mode": self.intent_population_mode,
            "source_manifest_supplied": self.source_manifest_supplied,
            "require_materialized_external": self.require_materialized_external,
            "requested_stage_count": sum(1 for stage in self.stages if stage.requested),
            "stages": [stage.as_manifest() for stage in self.stages],
            "by_stage": {stage.stage_id: stage.as_manifest() for stage in self.stages},
            "skip_reasons": self.skip_reason_map(),
        }


def _intent_value(intent: Any, name: str, default: Any = None) -> Any:
    if isinstance(intent, Mapping):
        return intent.get(name, default)
    return getattr(intent, name, default)


def _lower_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values.lower()}
    try:
        return {str(value).lower() for value in values}
    except TypeError:
        return {str(values).lower()}


def _contains_any(values: Iterable[str], needles: Iterable[str]) -> bool:
    haystack = {str(value).lower() for value in values}
    return any(any(needle in value for value in haystack) for needle in needles)


def _stage(
    *,
    stage_id: str,
    requested: bool,
    request_source: str,
    required_for_level3: bool,
    can_execute: bool,
    skip_reason: str,
    blocked_reason: str | None = None,
    executor: str | None = None,
    expected_artifacts: Iterable[str] = (),
) -> CompilerStageRequirement:
    return CompilerStageRequirement(
        stage_id=stage_id,
        requested=requested,
        request_source=request_source,
        required_for_level3=required_for_level3,
        can_execute=can_execute,
        skip_allowed=not requested,
        skip_reason=None if requested else skip_reason,
        blocked_reason=blocked_reason,
        executor=executor,
        expected_artifacts=tuple(expected_artifacts),
    )


def build_compile_stage_plan(
    *,
    intent: Any,
    population_tensor_mode: str | None = None,
    include_cnes_sih: bool = False,
    source_manifest: str | None = None,
    require_materialized_external: bool = False,
) -> CompilerStagePlan:
    """Build the proof-carrying stage contract for one compile run."""
    context_policy = _lower_set(_intent_value(intent, "context_policy", []))
    health_seeds = _lower_set(_intent_value(intent, "health_seeds", []))
    mandatory_fields = _lower_set(_intent_value(intent, "mandatory_fields", []))
    budget = _intent_value(intent, "budget", None)
    geo_mode = str(_intent_value(intent, "geo_mode", "native") or "native")
    run_profile = str(_intent_value(intent, "run_profile", "core_vital") or "core_vital")
    population_mode = str(_intent_value(intent, "population_mode", "") or "")
    tensor_mode = population_tensor_mode or None
    population_requested = bool(
        tensor_mode
        or population_mode in {"independent_population_tensor", "sim_informed_population_tensor", "independent_denominator", "sim_informed_denominator"}
        or "include_population_tensor" in context_policy
    )
    stdfm_requested = bool(
        run_profile in {"contextual", "full"}
        or
        _contains_any(context_policy, ("stdfm", "latent", "sidra_context"))
        or _contains_any(mandatory_fields, ("stdfm", "latent"))
    )
    pirs_model_requested = bool(
        _contains_any(context_policy, ("pirs", "residual", "model_execution", "run_pirs"))
        or _contains_any(mandatory_fields, ("residual", "modelassociation", "pirs"))
    )
    pirs_hsic_requested = bool(
        _contains_any(context_policy, ("hsic", "residual_scan", "hypothesis"))
        or _contains_any(mandatory_fields, ("hsic", "hypothesis"))
    )
    geo_requested = geo_mode.lower() not in {"native", "none", ""}
    stages = (
        _stage(
            stage_id="geo_support",
            requested=geo_requested,
            request_source="intent.geo_mode" if geo_requested else "native_geo_mode",
            required_for_level3=geo_requested,
            can_execute=geo_requested,
            skip_reason="intent.geo_mode is native; no AMC/geneallocation support transform requested",
            executor="pegasus.geo.support" if geo_requested else None,
            expected_artifacts=("geospatial_transform_manifest",) if geo_requested else (),
        ),
        _stage(
            stage_id="population_solver",
            requested=population_requested,
            request_source="intent.population_mode/context_policy" if population_requested else "official_sidra_anchor_or_unrequested",
            required_for_level3=population_requested,
            can_execute=population_requested,
            skip_reason="intent does not request population tensor optimization",
            executor="pegasus.sidra.population_cube.build.solve_population_tensor_from_sidra_strata" if population_requested else None,
            expected_artifacts=("Tables/population_tensor.parquet", "Tables/population_tensor_manifest.json") if population_requested else (),
        ),
        _stage(
            stage_id="stdfm",
            requested=stdfm_requested,
            request_source="intent.run_profile" if run_profile in {"contextual", "full"} else ("intent.context_policy/mandatory_fields" if stdfm_requested else "latent_context_not_requested"),
            required_for_level3=stdfm_requested,
            can_execute=stdfm_requested,
            skip_reason="intent does not request ST-DFM latent-factor fitting",
            executor="pegasus.she.stdfm.torch_solver.solve_stdfm" if stdfm_requested else None,
            expected_artifacts=("stage_workspace/stdfm/stdfm_latent_factors.parquet", "stage_workspace/stdfm/stdfm_certification.parquet") if stdfm_requested else (),
        ),
        _stage(
            stage_id="pirs_model",
            requested=pirs_model_requested,
            request_source="intent.context_policy/mandatory_fields" if pirs_model_requested else "pirs_model_not_requested",
            required_for_level3=pirs_model_requested,
            can_execute=pirs_model_requested,
            skip_reason="intent does not request PIRS model execution",
            executor="pegasus.pirs.model_execution.execute_pirs_model_from_design_matrix" if pirs_model_requested else None,
            expected_artifacts=("Tables/pirs_model_execution_manifest.json", "Tables/pirs_residual_values.parquet") if pirs_model_requested else (),
        ),
        _stage(
            stage_id="pirs_hsic",
            requested=pirs_hsic_requested,
            request_source="intent.context_policy/mandatory_fields" if pirs_hsic_requested else "pirs_hsic_not_requested",
            required_for_level3=pirs_hsic_requested,
            can_execute=pirs_hsic_requested,
            skip_reason="intent does not request HSIC residual scan",
            executor="pegasus.pirs.hsic_run.execute_hsic_residual_scan" if pirs_hsic_requested else None,
            expected_artifacts=("Tables/hsic_residual_scan_manifest.json",) if pirs_hsic_requested else (),
        ),
    )
    return CompilerStagePlan(
        schema_version="26B.1",
        plan_id="compile_stage_plan_v1",
        intent_budget=None if budget is None else str(budget),
        intent_geo_mode=geo_mode,
        intent_population_mode=population_mode or tensor_mode,
        source_manifest_supplied=bool(source_manifest),
        require_materialized_external=require_materialized_external,
        stages=stages,
    )


def normalize_stage_plan_manifest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    by_stage = payload.get("by_stage")
    if isinstance(by_stage, Mapping):
        return {"by_stage": dict(by_stage), "raw": dict(payload)}
    stages = payload.get("stages")
    if isinstance(stages, list):
        return {"by_stage": {str(stage.get("stage_id")): dict(stage) for stage in stages if isinstance(stage, Mapping)}, "raw": dict(payload)}
    return {"by_stage": {}, "raw": dict(payload)}


def validate_compiler_stage_plan(
    *,
    stage_plan: Mapping[str, Any] | None,
    stage_status: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return structural errors for telemetry/status consistency."""
    normalized = normalize_stage_plan_manifest(stage_plan)
    by_stage = normalized.get("by_stage", {})
    if not by_stage:
        return ("compiler_stage_plan_missing",)
    statuses = {str(key): str(value) for key, value in (stage_status or {}).items()}
    errors: list[str] = []
    for stage_id, spec in by_stage.items():
        if not stage_id:
            continue
        requested = bool(spec.get("requested"))
        skip_allowed = bool(spec.get("skip_allowed"))
        skip_reason = spec.get("skip_reason")
        status = statuses.get(stage_id)
        if status is None:
            errors.append(f"stage_status_missing:{stage_id}")
            continue
        if status == "skipped" and requested:
            errors.append(f"requested_stage_skipped:{stage_id}")
        if status == "skipped" and not skip_allowed:
            errors.append(f"skip_not_allowed:{stage_id}")
        if status == "skipped" and not skip_reason:
            errors.append(f"skip_reason_missing:{stage_id}")
        if status == "success" and not requested and spec.get("request_source") not in {"native_geo_mode", "official_sidra_anchor_or_unrequested"}:
            # Non-requested success can happen for base compile stages outside this plan;
            # for optional stages in this plan it is suspicious but not fatal.
            pass
    return tuple(errors)
