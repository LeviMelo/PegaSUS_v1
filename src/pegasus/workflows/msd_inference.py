"""MSD inference-stage orchestration for completed compile run bundles.

This module wires the PIRS model and HSIC residual-scan stages into the
compile pipeline. It is deliberately defensive: requested inference stages
are attempted; if a readiness gate blocks them, telemetry records `blocked`
with a reason rather than silently reporting `skipped`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from pegasus.output.reproducibility import RunTelemetry
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.pirs_pipeline import run_pirs_planning_pipeline
from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run
from pegasus.workflows.pirs_execute import run_execute_pirs_model
from pegasus.workflows.hsic_execute import run_execute_hsic_residual_scan
from pegasus.workflows.hsic_rank import run_attach_hsic_ranking_to_run
from pegasus.workflows.hsic_report import run_attach_hsic_report_to_run


def _stage_manifest(stage_plan: Any) -> dict[str, Any]:
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _attach_summary(root: Path, payload: dict[str, Any]) -> Path:
    out = root / "Tables" / "msd_inference_pipeline.json"
    _write_json(out, payload)
    for rel in ("RunConfig.json", "P_vector.json", "ReproducibilityManifest.json"):
        path = root / rel
        current = _load_json(path)
        if not current:
            continue
        current["msd_inference_pipeline"] = {
            "manifest_path": str(out),
            "pirs_model_status": payload.get("pirs_model", {}).get("status"),
            "pirs_hsic_status": payload.get("pirs_hsic", {}).get("status"),
        }
        _write_json(path, current)
    return out


def _record_stage(
    telemetry: RunTelemetry,
    *,
    stage_id: str,
    status: str,
    elapsed: float,
    reason: str | None = None,
) -> None:
    telemetry.set_stage(stage_id, status, elapsed)
    if reason:
        telemetry.resource_summary.setdefault("stage_errors", {})[stage_id] = reason


def run_msd_inference_pipeline(
    *,
    run_dir: str | Path,
    compiler_stage_plan: Any,
    telemetry: RunTelemetry,
    budget: str = "fast",
) -> dict[str, Any]:
    root = Path(run_dir)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline": "msd_inference_pipeline",
        "budget": budget,
        "pirs_model": {"requested": _requested(compiler_stage_plan, "pirs_model")},
        "pirs_hsic": {"requested": _requested(compiler_stage_plan, "pirs_hsic")},
        "artifacts": {},
    }

    model_success = False
    design_matrix_manifest: dict[str, Any] | str | None = None
    model_execution_manifest: dict[str, Any] | str | None = None

    if not payload["pirs_model"]["requested"]:
        _record_stage(
            telemetry,
            stage_id="pirs_model",
            status="skipped",
            elapsed=0.0,
            reason="intent does not request PIRS model execution",
        )
        payload["pirs_model"].update({"status": "skipped", "reason": "not_requested"})
    else:
        started = time.perf_counter()
        try:
            planning = run_pirs_planning_pipeline(run_dir=root, budget=budget)
            pipeline = planning.get("pipeline", {})
            summary = planning.get("pirs_planning_pipeline_gate", {})
            payload["artifacts"]["pirs_planning_pipeline"] = planning.get("manifest_path")

            if summary.get("status") not in {"ready", "success"}:
                reason = str(summary.get("reason") or summary.get("status") or "pirs_design_not_ready")
                elapsed = time.perf_counter() - started
                _record_stage(telemetry, stage_id="pirs_model", status="blocked", elapsed=elapsed, reason=reason)
                payload["pirs_model"].update({"status": "blocked", "reason": reason, "planning": summary})
            else:
                design_plan = (pipeline.get("artifacts") or {}).get("pirs_design_plan")
                readiness = (pipeline.get("artifacts") or {}).get("pirs_design_readiness")
                matrix = run_attach_pirs_design_matrix_to_run(
                    run_dir=root,
                    design_plan=design_plan,
                    readiness_manifest=readiness,
                )
                design_matrix_manifest = matrix
                payload["artifacts"]["pirs_design_matrix"] = matrix.get("manifest_path") or matrix.get("design_matrix_manifest")

                model = run_execute_pirs_model(
                    run_dir=root,
                    design_matrix_manifest=matrix,
                    mutate_output_bundle=True,
                    validate=True,
                    attach=True,
                )
                model_execution_manifest = model
                payload["artifacts"]["pirs_model_execution"] = model.get("manifest_path") or model.get("model_execution_manifest")
                elapsed = time.perf_counter() - started
                _record_stage(telemetry, stage_id="pirs_model", status="success", elapsed=elapsed)
                payload["pirs_model"].update({"status": "success", "planning": summary, "execution": model})
                model_success = True
        except Exception as exc:
            elapsed = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            _record_stage(telemetry, stage_id="pirs_model", status="blocked", elapsed=elapsed, reason=reason)
            payload["pirs_model"].update({"status": "blocked", "reason": reason})

    if not payload["pirs_hsic"]["requested"]:
        _record_stage(
            telemetry,
            stage_id="pirs_hsic",
            status="skipped",
            elapsed=0.0,
            reason="intent does not request HSIC residual scan",
        )
        payload["pirs_hsic"].update({"status": "skipped", "reason": "not_requested"})
    elif not model_success:
        reason = "pirs_model_not_successful"
        _record_stage(telemetry, stage_id="pirs_hsic", status="blocked", elapsed=0.0, reason=reason)
        payload["pirs_hsic"].update({"status": "blocked", "reason": reason})
    else:
        started = time.perf_counter()
        try:
            scan = run_execute_hsic_residual_scan(
                run_dir=root,
                model_execution_manifest=model_execution_manifest,
                design_matrix_manifest=design_matrix_manifest,
                budget=budget,
                mutate_output_bundle=True,
                validate=True,
                attach=True,
            )
            payload["artifacts"]["hsic_residual_scan"] = scan.get("manifest_path") or scan.get("output_manifest")
            try:
                ranking = run_attach_hsic_ranking_to_run(run_dir=root)
                payload["artifacts"]["hsic_ranking"] = ranking.get("manifest_path") or ranking.get("ranking_manifest")
            except Exception as rank_exc:
                payload.setdefault("warnings", []).append(f"hsic_ranking_not_attached:{type(rank_exc).__name__}:{rank_exc}")
            try:
                report = run_attach_hsic_report_to_run(run_dir=root)
                payload["artifacts"]["hsic_report"] = report.get("manifest_path") or report.get("report_manifest")
            except Exception as report_exc:
                payload.setdefault("warnings", []).append(f"hsic_report_not_attached:{type(report_exc).__name__}:{report_exc}")

            elapsed = time.perf_counter() - started
            _record_stage(telemetry, stage_id="pirs_hsic", status="success", elapsed=elapsed)
            payload["pirs_hsic"].update({"status": "success", "scan": scan})
        except Exception as exc:
            elapsed = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            _record_stage(telemetry, stage_id="pirs_hsic", status="blocked", elapsed=elapsed, reason=reason)
            payload["pirs_hsic"].update({"status": "blocked", "reason": reason})

    manifest_path = _attach_summary(root, payload)
    payload["manifest_path"] = str(manifest_path)

    validation = validate_output_bundle(run_dir=str(root))
    payload["output_validation"] = {"ok": bool(validation.ok), "errors": list(validation.errors)}
    return payload


__all__ = ["run_msd_inference_pipeline"]
