"""MSD inference phase (PIRS model fit + residual + HSIC scan) — typed driver.

This replaces the previous ``importlib``-based string dispatch (which guessed
function names from tuples and wrapped everything in ``try/except`` that relabeled
real execution failures as benign "blocked" stages) with direct, typed calls and
the honesty rails from MSD §10: an execution error is a *failure* that aborts the
compile, never a silently swallowed skip.

The planning chain (candidate → selection → design plan → readiness) is inlined
here rather than going through the old "intentionally non-executing" planning
wrapper. State flows through a :class:`RunContext` and in-memory dicts; stages
still emit their first-class artifacts for the immutable output bundle.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from pegasus.core.run_context import RunContext

# Direct, typed stage imports — no string dispatch.
from pegasus.workflows.pirs_candidates import run_attach_pirs_candidate_gate
from pegasus.workflows.pirs_selection import run_attach_pirs_selection_plan
from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run
from pegasus.workflows.pirs_readiness import run_attach_pirs_design_readiness_to_run
from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run
from pegasus.workflows.pirs_execute import run_execute_pirs_model
from pegasus.workflows.hsic_execute import run_execute_hsic_residual_scan
from pegasus.workflows.hsic_rank import run_attach_hsic_ranking_to_run
from pegasus.workflows.hsic_report import run_attach_hsic_report_to_run


def _stage_manifest(stage_plan: Any) -> dict[str, dict[str, Any]]:
    if stage_plan is None:
        return {}
    if hasattr(stage_plan, "as_manifest"):
        payload = stage_plan.as_manifest()
    elif isinstance(stage_plan, Mapping):
        payload = dict(stage_plan)
    else:
        return {}
    by_stage = payload.get("by_stage")
    if isinstance(by_stage, Mapping):
        return {str(k): dict(v) for k, v in by_stage.items() if isinstance(v, Mapping)}
    stages = payload.get("stages")
    if isinstance(stages, list):
        return {
            str(stage.get("stage_id")): dict(stage)
            for stage in stages
            if isinstance(stage, Mapping) and stage.get("stage_id")
        }
    return {}


def _requested(stage_plan: Any, stage_id: str) -> bool:
    spec = _stage_manifest(stage_plan).get(stage_id)
    return bool(spec and spec.get("requested"))


def _record_stage(telemetry: Any, *, stage_id: str, status: str, elapsed: float, reason: str | None = None) -> None:
    telemetry.set_stage(stage_id, status, elapsed)
    if reason:
        telemetry.resource_summary.setdefault("stage_errors", {})[stage_id] = reason


def _write_stage_manifest(root: Path, name: str, payload: dict[str, Any]) -> Path:
    out = root / "Tables" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return out


def _status_of(payload: Any, *keys: str) -> str:
    """Pull a status string from a payload or a nested gate/summary."""
    if isinstance(payload, Mapping):
        for key in ("status", *keys):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        for nested in ("summary", "gate", "design_matrix", "model_execution"):
            value = payload.get(nested)
            if isinstance(value, Mapping) and isinstance(value.get("status"), str):
                return str(value["status"])
    return "unknown"


def _blocking_reason(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        for key in ("reason", "blocked_reason"):
            if isinstance(payload.get(key), str):
                return payload[key]
        reasons = payload.get("blocking_reasons")
        if isinstance(reasons, list) and reasons:
            return "; ".join(str(r) for r in reasons)
        nested = payload.get("design_matrix")
        if isinstance(nested, Mapping):
            return _blocking_reason(nested)
    return None


def _run_pirs_model_phase(ctx: RunContext, payload: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
    """Planning + design matrix + model fit. Returns (success, matrix, model_exec)."""
    pirs_root = ctx.pirs_root
    budget = ctx.budget

    # Planning chain (inlined, direct calls). Each writes its first-class gate
    # artifact; "blocked" here is a legitimate MSD-declared state, not a bug.
    run_attach_pirs_candidate_gate(run_dir=pirs_root)
    run_attach_pirs_selection_plan(run_dir=pirs_root, budget=budget)
    run_attach_pirs_design_plan_to_run(run_dir=pirs_root, budget=budget)
    run_attach_pirs_design_readiness_to_run(run_dir=pirs_root)

    matrix = run_attach_pirs_design_matrix_to_run(run_dir=pirs_root)
    matrix_payload = matrix.get("design_matrix") if isinstance(matrix, Mapping) else None
    payload["artifacts"]["pirs_design_matrix"] = (matrix or {}).get("manifest_path")
    if _status_of(matrix_payload or matrix) != "ready":
        reason = _blocking_reason(matrix) or "design_matrix_not_ready"
        payload["pirs_model"].update({"status": "blocked", "reason": reason})
        return False, matrix_payload, None

    model = run_execute_pirs_model(
        run_dir=pirs_root,
        design_matrix_manifest=matrix_payload,
        mutate_output_bundle=True,
        validate=False,
        attach=True,
        bundle=ctx.bundle,
    )
    model_exec = model.get("model_execution") if isinstance(model, Mapping) else None
    payload["artifacts"]["pirs_model_execution"] = (model or {}).get("manifest_path")
    if _status_of(model_exec or model) != "fitted":
        reason = _blocking_reason(model) or "model_not_fitted"
        payload["pirs_model"].update({"status": "blocked", "reason": reason})
        return False, matrix_payload, model_exec
    payload["pirs_model"].update({"status": "success", "execution": model_exec})
    return True, matrix_payload, model_exec


def _run_hsic_phase(
    ctx: RunContext,
    payload: dict[str, Any],
    *,
    matrix_payload: dict[str, Any] | None,
    model_exec: dict[str, Any] | None,
) -> None:
    pirs_root = ctx.pirs_root
    scan = run_execute_hsic_residual_scan(
        run_dir=pirs_root,
        model_execution_manifest=model_exec,
        design_matrix_manifest=matrix_payload,
        budget=ctx.budget,
        mutate_output_bundle=True,
        validate=False,
        attach=True,
        bundle=ctx.bundle,
    )
    payload["artifacts"]["hsic_residual_scan"] = (scan or {}).get("manifest_path") if isinstance(scan, Mapping) else None
    for attach_fn, key in ((run_attach_hsic_ranking_to_run, "hsic_ranking"), (run_attach_hsic_report_to_run, "hsic_report")):
        try:
            result = attach_fn(run_dir=pirs_root)
            payload["artifacts"][key] = result.get("manifest_path") if isinstance(result, Mapping) else None
        except Exception as exc:  # ranking/report are non-essential annotations
            payload.setdefault("warnings", []).append(f"{key}_not_attached:{type(exc).__name__}:{exc}")
    payload["pirs_hsic"].update({"status": "success", "scan": scan})


def run_msd_inference_pipeline(
    *,
    run_dir: str | Path,
    compiler_stage_plan: Any,
    telemetry: Any,
    budget: str = "fast",
    bundle: Any | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(
        run_dir=root,
        budget=budget,
        bundle=bundle,
        telemetry=telemetry,
        compiler_stage_plan=compiler_stage_plan,
    )
    ctx.stage_pirs_workspace()

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline": "msd_inference_pipeline",
        "budget": budget,
        "pirs_model": {"requested": _requested(compiler_stage_plan, "pirs_model")},
        "pirs_hsic": {"requested": _requested(compiler_stage_plan, "pirs_hsic")},
        "artifacts": {},
    }

    model_success = False
    matrix_payload: dict[str, Any] | None = None
    model_exec: dict[str, Any] | None = None

    if not payload["pirs_model"]["requested"]:
        _record_stage(telemetry, stage_id="pirs_model", status="skipped", elapsed=0.0, reason="intent does not request PIRS model execution")
        payload["pirs_model"].update({"status": "skipped", "reason": "not_requested"})
    else:
        started = time.perf_counter()
        try:
            model_success, matrix_payload, model_exec = _run_pirs_model_phase(ctx, payload)
            elapsed = time.perf_counter() - started
            _record_stage(
                telemetry,
                stage_id="pirs_model",
                status="success" if model_success else "blocked",
                elapsed=elapsed,
                reason=None if model_success else payload["pirs_model"].get("reason"),
            )
        except Exception as exc:
            # Honesty rail (MSD §10): execution errors are failures, not blocks.
            elapsed = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            _record_stage(telemetry, stage_id="pirs_model", status="failed", elapsed=elapsed, reason=reason)
            payload["pirs_model"].update({"status": "failed", "reason": reason})
            telemetry.flush()
            raise

    if not payload["pirs_hsic"]["requested"]:
        _record_stage(telemetry, stage_id="pirs_hsic", status="skipped", elapsed=0.0, reason="intent does not request HSIC residual scan")
        payload["pirs_hsic"].update({"status": "skipped", "reason": "not_requested"})
    elif not model_success:
        _record_stage(telemetry, stage_id="pirs_hsic", status="blocked", elapsed=0.0, reason="pirs_model_not_successful")
        payload["pirs_hsic"].update({"status": "blocked", "reason": "pirs_model_not_successful"})
    else:
        started = time.perf_counter()
        try:
            _run_hsic_phase(ctx, payload, matrix_payload=matrix_payload, model_exec=model_exec)
            _record_stage(telemetry, stage_id="pirs_hsic", status="success", elapsed=time.perf_counter() - started)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            _record_stage(telemetry, stage_id="pirs_hsic", status="failed", elapsed=elapsed, reason=reason)
            payload["pirs_hsic"].update({"status": "failed", "reason": reason})
            telemetry.flush()
            raise

    manifest_path = _write_stage_manifest(ctx.pirs_root, "msd_inference_pipeline.json", payload)
    try:
        payload["manifest_path"] = str(manifest_path.relative_to(ctx.pirs_root)).replace("\\", "/")
    except ValueError:
        payload["manifest_path"] = str(manifest_path)
    payload["output_validation"] = {"status": "deferred_until_output_bundle_flush"}
    if bundle is not None:
        bundle.set_artifact_dir("Tables", ctx.pirs_root / "Tables")
    return payload


__all__ = ["run_msd_inference_pipeline"]
