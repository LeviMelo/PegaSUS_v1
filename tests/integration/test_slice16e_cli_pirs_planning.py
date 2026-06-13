
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pegasus.cli import app


runner = CliRunner()


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_slice16e_cli_inspects_planning_artifacts_read_only(tmp_path: Path) -> None:
    candidates = tmp_path / "pirs_field_candidates.json"
    selection = tmp_path / "pirs_selection_plan.json"
    design = tmp_path / "pirs_design_plan.json"
    readiness = tmp_path / "pirs_design_readiness.json"

    _write(candidates, {"gate": "pirs_candidate_gate", "candidate_count": 2, "rejected_count": 1, "candidates": [{}, {}], "rejected": [{}]})
    _write(selection, {"artifact": "pirs_selection_plan", "status": "planned", "budget": "fast", "selected_outcome_field_id": "field:outcome", "selected_covariate_field_ids": ["field:cov"], "selection_rejected": []})
    _write(design, {"artifact": "pirs_design_plan", "status": "planned", "design_matrix_state": "planned_only", "model_fit_state": "not_started", "residual_state": "not_started", "terms": [{}, {}]})
    _write(readiness, {"artifact": "pirs_design_readiness_gate", "status": "blocked", "ready": False, "accepted_fields": [], "rejected_fields": [{"field_id": "x"}]})

    result = runner.invoke(app, ["pirs", "inspect-candidates", "--manifest", str(candidates)])
    assert result.exit_code == 0, result.output
    assert '"candidate_count": 2' in result.output
    assert '"read_only": true' in result.output

    result = runner.invoke(app, ["pirs", "inspect-selection-plan", "--plan", str(selection)])
    assert result.exit_code == 0, result.output
    assert '"selected_covariate_count": 1' in result.output

    result = runner.invoke(app, ["pirs", "inspect-design-plan", "--plan", str(design)])
    assert result.exit_code == 0, result.output
    assert '"design_matrix_state": "planned_only"' in result.output

    result = runner.invoke(app, ["pirs", "inspect-design-readiness", "--manifest", str(readiness)])
    assert result.exit_code == 0, result.output
    assert '"ready": false' in result.output
    assert '"rejected_field_count": 1' in result.output


def test_slice16e_cli_help_lists_pirs_planning_commands() -> None:
    result = runner.invoke(app, ["pirs", "--help"])
    assert result.exit_code == 0, result.output
    for command in (
        "candidates-from-run",
        "attach-candidate-gate",
        "plan-selection",
        "attach-selection-plan",
        "plan-design",
        "attach-design-plan",
        "check-design-readiness",
        "attach-design-readiness",
    ):
        assert command in result.output
