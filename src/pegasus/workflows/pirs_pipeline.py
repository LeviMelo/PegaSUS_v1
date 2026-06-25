
"""PIRS planning-pipeline orchestrator for run bundles.

Slice 16F composes the existing 16A-16D workflow services.  It is still a
planning boundary: it does not fit models, build numerical design matrices,
create residuals, run HSIC, or mutate compile outputs beyond attaching a
compact summary to existing JSON surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.workflows.pirs_candidates import run_attach_pirs_candidate_gate
from pegasus.workflows.pirs_selection import run_attach_pirs_selection_plan
from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run
from pegasus.workflows.pirs_readiness import run_attach_pirs_design_readiness_to_run


DEFAULT_PIPELINE_MANIFEST = Path("Tables") / "pirs_planning_pipeline.json"
JSON_SURFACES: tuple[str, ...] = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path


def _gate_from_payload(payload: Any, key: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        gate = payload.get(key)
        if isinstance(gate, dict):
            return gate
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return summary
    return {}


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _manifest_path(value: dict[str, Any], fallback: Path) -> str:
    path = value.get("manifest_path")
    if path is not None:
        return str(path)
    return str(fallback)



def pirs_planning_pipeline_summary(payload: Any, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    payload_map = _as_mapping(payload)
    candidate_gate = _as_mapping(payload_map.get("candidate_gate"))
    selection_gate = _as_mapping(payload_map.get("selection_gate"))
    design_gate = _as_mapping(payload_map.get("design_gate"))
    readiness_gate = _as_mapping(payload_map.get("design_readiness_gate"))

    return {
        "schema_version": "1.0",
        "slice": "16F",
        "gate": "pirs_planning_pipeline_gate",
        "status": str(payload_map.get("status", "blocked")),
        "ready": bool(readiness_gate.get("ready", False)),
        "budget": payload_map.get("budget"),
        "candidate_count": candidate_gate.get("candidate_count"),
        "candidate_rejected_count": candidate_gate.get("rejected_count"),
        "selection_status": selection_gate.get("status"),
        "selected_outcome_field_id": selection_gate.get("selected_outcome_field_id"),
        "selected_covariate_count": selection_gate.get("selected_covariate_count"),
        "design_status": design_gate.get("status"),
        "design_matrix_state": readiness_gate.get("design_matrix_state", design_gate.get("design_matrix_state")),
        "readiness_status": readiness_gate.get("status"),
        "ready_field_count": readiness_gate.get("ready_field_count"),
        "blocked_field_count": readiness_gate.get("blocked_field_count"),
        "model_fit_state": "not_started",
        "residual_state": "not_started",
        "hsic_state": "not_started",
        "manifest_path": str(manifest_path) if manifest_path is not None else payload_map.get("manifest_path"),
        "non_mutating_planning_only": True,
    }


def attach_pirs_planning_pipeline_summary_to_run(
    *,
    run_dir: str | Path,
    summary: dict[str, Any],
) -> None:
    root = Path(run_dir)
    for rel in JSON_SURFACES:
        path = root / rel
        if not path.exists():
            continue
        payload = _load_json(path)
        payload["pirs_planning_pipeline_gate"] = summary
        _write_json(path, payload)


def run_pirs_planning_pipeline(
    *,
    run_dir: str | Path,
    budget: str = "fast",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the 16A-16D PIRS planning chain and attach a compact summary.

    The chain is intentionally non-executing.  It stops at design readiness
    and records whether a future numerical design-matrix builder may proceed.
    """
    root = Path(run_dir)
    out = Path(output) if output is not None else root / DEFAULT_PIPELINE_MANIFEST

    candidate_gate = run_attach_pirs_candidate_gate(run_dir=root)
    selection_gate = run_attach_pirs_selection_plan(run_dir=root, budget=budget)
    design_payload = run_attach_pirs_design_plan_to_run(run_dir=root, budget=budget)
    design_gate = _gate_from_payload(design_payload, "pirs_design_gate")
    readiness_payload = run_attach_pirs_design_readiness_to_run(run_dir=root)
    readiness_gate = _gate_from_payload(readiness_payload, "pirs_design_readiness_gate")

    status = "ready" if readiness_gate.get("status") == "ready" else "blocked"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "slice": "16F",
        "pipeline": "pirs_planning_pipeline",
        "status": status,
        "budget": budget,
        "candidate_gate": candidate_gate,
        "selection_gate": selection_gate,
        "design_gate": design_gate,
        "design_readiness_gate": readiness_gate,
        "artifacts": {
            "pirs_field_candidates": _manifest_path(candidate_gate, root / "Tables" / "pirs_field_candidates.json"),
            "pirs_selection_plan": _manifest_path(selection_gate, root / "Tables" / "pirs_selection_plan.json"),
            "pirs_design_plan": _manifest_path(design_gate, root / "Tables" / "pirs_design_plan.json"),
            "pirs_design_readiness": _manifest_path(readiness_gate, root / "Tables" / "pirs_design_readiness.json"),
        },
        "model_execution_state": "not_started",
        "residual_state": "not_started",
        "hsic_state": "not_started",
        "non_mutating_planning_only": True,
    }
    _write_json(out, payload)
    summary = pirs_planning_pipeline_summary(payload, manifest_path=out)
    payload["summary"] = summary
    payload["manifest_path"] = str(out)
    _write_json(out, payload)
    attach_pirs_planning_pipeline_summary_to_run(run_dir=root, summary=summary)
    return {"manifest_path": str(out), "pirs_planning_pipeline_gate": summary, "pipeline": payload}


def inspect_pirs_planning_pipeline_manifest(manifest: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(manifest))
    if not payload:
        raise FileNotFoundError(f"missing or invalid PIRS planning pipeline manifest: {manifest}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return pirs_planning_pipeline_summary(payload, manifest_path=manifest)
