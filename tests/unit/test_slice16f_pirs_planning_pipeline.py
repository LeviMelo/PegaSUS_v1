
from __future__ import annotations

import json
from pathlib import Path

from pegasus.workflows import pirs_pipeline


def _json_surfaces(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "Tables").mkdir(exist_ok=True)


def test_slice16f_pipeline_composes_gates_and_attaches_summary(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    _json_surfaces(run_dir)

    monkeypatch.setattr(
        pirs_pipeline,
        "run_attach_pirs_candidate_gate",
        lambda **kwargs: {"status": "evaluated", "candidate_count": 2, "rejected_count": 1, "manifest_path": str(run_dir / "Tables" / "pirs_field_candidates.json")},
    )
    monkeypatch.setattr(
        pirs_pipeline,
        "run_attach_pirs_selection_plan",
        lambda **kwargs: {"status": "planned", "selected_outcome_field_id": "outcome", "selected_covariate_count": 1, "manifest_path": str(run_dir / "Tables" / "pirs_selection_plan.json")},
    )
    monkeypatch.setattr(
        pirs_pipeline,
        "run_attach_pirs_design_plan_to_run",
        lambda **kwargs: {"pirs_design_gate": {"status": "planned", "design_matrix_state": "planned_only", "manifest_path": str(run_dir / "Tables" / "pirs_design_plan.json")}},
    )
    monkeypatch.setattr(
        pirs_pipeline,
        "run_attach_pirs_design_readiness_to_run",
        lambda **kwargs: {"pirs_design_readiness_gate": {"status": "blocked", "ready": False, "ready_field_count": 0, "blocked_field_count": 2, "design_matrix_state": "blocked_until_tensor_backed_fields", "manifest_path": str(run_dir / "Tables" / "pirs_design_readiness.json")}},
    )

    result = pirs_pipeline.run_pirs_planning_pipeline(run_dir=run_dir, budget="standard")
    gate = result["pirs_planning_pipeline_gate"]
    assert gate["status"] == "blocked"
    assert gate["budget"] == "standard"
    assert gate["candidate_count"] == 2
    assert gate["selected_outcome_field_id"] == "outcome"
    assert gate["blocked_field_count"] == 2
    assert gate["model_fit_state"] == "not_started"
    assert Path(result["manifest_path"]).exists()
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["pirs_planning_pipeline_gate"]["status"] == "blocked"
