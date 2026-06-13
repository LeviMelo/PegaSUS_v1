
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.pirs_pipeline import inspect_pirs_planning_pipeline_manifest, run_pirs_planning_pipeline


def test_slice16f_pipeline_runs_on_empty_bundle_as_blocked_planning_chain(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)

    result = run_pirs_planning_pipeline(run_dir=run_dir, budget="fast")
    gate = result["pirs_planning_pipeline_gate"]
    assert gate["status"] == "blocked"
    assert gate["ready"] is False
    assert gate["model_fit_state"] == "not_started"
    assert gate["residual_state"] == "not_started"
    assert gate["hsic_state"] == "not_started"
    assert (run_dir / "Tables" / "pirs_field_candidates.json").exists()
    assert (run_dir / "Tables" / "pirs_selection_plan.json").exists()
    assert (run_dir / "Tables" / "pirs_design_plan.json").exists()
    assert (run_dir / "Tables" / "pirs_design_readiness.json").exists()
    assert (run_dir / "Tables" / "pirs_planning_pipeline.json").exists()

    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["pirs_planning_pipeline_gate"]["status"] == "blocked"
    inspected = inspect_pirs_planning_pipeline_manifest(run_dir / "Tables" / "pirs_planning_pipeline.json")
    assert inspected["status"] == "blocked"
