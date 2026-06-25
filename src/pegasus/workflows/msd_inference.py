from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq

from pegasus.output.validate import validate_output_bundle


FIRST_CLASS_INFERENCE_TABLES: dict[str, str] = {
    "ModelAssociations": "ModelAssociations.parquet",
    "ResidualAssociations": "ResidualAssociations.parquet",
    "Hypotheses": "Hypotheses.parquet",
}


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


def _load_callable(module_name: str, *names: str) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise AttributeError(f"{module_name} does not expose any of {names!r}")


def _write_stage_manifest(root: Path, name: str, payload: dict[str, Any]) -> Path:
    out = root / "Tables" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return out


def _stage_first_class_tables_from_run(bundle: Any | None, root: Path) -> None:
    if bundle is None:
        return
    for key, rel in FIRST_CLASS_INFERENCE_TABLES.items():
        path = root / rel
        if path.exists():
            bundle.set_table(key, pq.read_table(path).to_pylist())


def _record_stage(
    telemetry: Any,
    *,
    stage_id: str,
    status: str,
    elapsed: float,
    reason: str | None = None,
) -> None:
    telemetry.set_stage(stage_id, status, elapsed)
    if reason:
        telemetry.resource_summary.setdefault("stage_errors", {})[stage_id] = reason


def _pipeline_gate_status(planning: dict[str, Any]) -> tuple[str, str | None]:
    gate = planning.get("pirs_planning_pipeline_gate")
    if isinstance(gate, dict):
        status = str(gate.get("status") or "")
        reason = gate.get("reason") or gate.get("blocked_reason")
        if status in {"ready", "success"}:
            return "ready", None
        if status:
            return status, str(reason or status)
    pipeline = planning.get("pipeline")
    if isinstance(pipeline, dict):
        return "ready", None
    return "blocked", "pirs_planning_pipeline_did_not_return_ready_gate"


def _artifact(payload: Any, *keys: str) -> Any:
    if not isinstance(payload, Mapping):
        return payload
    for key in keys:
        if key in payload:
            return payload[key]
    return payload




def _prepare_pirs_workspace(bundle: Any | None, root: Path) -> Path:
    if bundle is None:
        return root
    workspace = root.parent / f"{root.name}__pirs_stage_workspace"
    return bundle.write_stage_workspace(workspace)



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

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline": "msd_inference_pipeline",
        "budget": budget,
        "pirs_model": {"requested": _requested(compiler_stage_plan, "pirs_model")},
        "pirs_hsic": {"requested": _requested(compiler_stage_plan, "pirs_hsic")},
        "artifacts": {},
    }

    pirs_root = _prepare_pirs_workspace(bundle, root)
    model_success = False
    design_matrix_manifest: Any = None
    model_execution_manifest: Any = None

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
            run_planning = _load_callable(
                "pegasus.workflows.pirs_pipeline",
                "run_pirs_planning_pipeline",
                "run_attach_pirs_planning_pipeline_to_run",
            )
            planning = run_planning(run_dir=pirs_root, budget=budget)
            payload["artifacts"]["pirs_planning_pipeline"] = _artifact(planning, "manifest_path")
            status, reason = _pipeline_gate_status(planning)

            if status != "ready":
                elapsed = time.perf_counter() - started
                _record_stage(telemetry, stage_id="pirs_model", status="blocked", elapsed=elapsed, reason=reason)
                payload["pirs_model"].update({"status": "blocked", "reason": reason, "planning": planning})
            else:
                pipeline = planning.get("pipeline", {}) if isinstance(planning, Mapping) else {}
                artifacts = pipeline.get("artifacts", {}) if isinstance(pipeline, Mapping) else {}
                design_plan = artifacts.get("pirs_design_plan")
                readiness = artifacts.get("pirs_design_readiness")

                run_matrix = _load_callable(
                    "pegasus.workflows.pirs_matrix",
                    "run_attach_pirs_design_matrix_to_run",
                    "run_pirs_design_matrix",
                )
                matrix = run_matrix(
                    run_dir=pirs_root,
                    design_plan=design_plan,
                    readiness_manifest=readiness,
                )
                design_matrix_manifest = matrix
                payload["artifacts"]["pirs_design_matrix"] = _artifact(
                    matrix,
                    "manifest_path",
                    "design_matrix_manifest",
                )

                run_model = _load_callable(
                    "pegasus.workflows.pirs_execute",
                    "run_execute_pirs_model",
                    "run_pirs_model",
                )
                model = run_model(
                    run_dir=pirs_root,
                    design_matrix_manifest=matrix,
                    mutate_output_bundle=True,
                    validate=True,
                    attach=True,
                    bundle=bundle,
                )
                model_execution_manifest = model
                payload["artifacts"]["pirs_model_execution"] = _artifact(
                    model,
                    "manifest_path",
                    "model_execution_manifest",
                )
                

                elapsed = time.perf_counter() - started
                _record_stage(telemetry, stage_id="pirs_model", status="success", elapsed=elapsed)
                payload["pirs_model"].update({"status": "success", "planning": planning, "execution": model})
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
            run_scan = _load_callable(
                "pegasus.workflows.hsic_execute",
                "run_execute_hsic_residual_scan",
                "run_hsic_residual_scan",
            )
            scan = run_scan(
                run_dir=pirs_root,
                model_execution_manifest=model_execution_manifest,
                design_matrix_manifest=design_matrix_manifest,
                budget=budget,
                mutate_output_bundle=True,
                validate=True,
                attach=True,
                bundle=bundle,
            )
            payload["artifacts"]["hsic_residual_scan"] = _artifact(scan, "manifest_path", "output_manifest")

            for module_name, names, key in (
                ("pegasus.workflows.hsic_rank", ("run_attach_hsic_ranking_to_run", "run_hsic_ranking"), "hsic_ranking"),
                ("pegasus.workflows.hsic_report", ("run_attach_hsic_report_to_run", "run_hsic_report"), "hsic_report"),
            ):
                try:
                    fn = _load_callable(module_name, *names)
                    result = fn(run_dir=pirs_root)
                    payload["artifacts"][key] = _artifact(result, "manifest_path", f"{key}_manifest")
                except Exception as exc:
                    payload.setdefault("warnings", []).append(f"{key}_not_attached:{type(exc).__name__}:{exc}")

            
            elapsed = time.perf_counter() - started
            _record_stage(telemetry, stage_id="pirs_hsic", status="success", elapsed=elapsed)
            payload["pirs_hsic"].update({"status": "success", "scan": scan})
        except Exception as exc:
            elapsed = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            _record_stage(telemetry, stage_id="pirs_hsic", status="blocked", elapsed=elapsed, reason=reason)
            payload["pirs_hsic"].update({"status": "blocked", "reason": reason})

    manifest_path = _write_stage_manifest(root, "msd_inference_pipeline.json", payload)
    payload["manifest_path"] = str(manifest_path.relative_to(root)).replace("\\", "/")

    try:
        validation = validate_output_bundle(run_dir=str(root))
        payload["output_validation"] = {"ok": bool(validation.ok), "errors": list(validation.errors)}
    except Exception as exc:
        payload["output_validation"] = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}

    return payload


__all__ = ["run_msd_inference_pipeline"]
