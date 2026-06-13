
from __future__ import annotations

import json
from pathlib import Path

from pegasus.pirs.selection_plan import build_pirs_selection_plan, load_pirs_candidate_manifest


def _candidate(field_id: str, role: str, utility: float) -> dict:
    return {
        "field_id": field_id,
        "role": role,
        "utility": utility,
        "q_state": "verified",
        "carrier": "Deaths",
        "unit": "counts",
        "support": {"years": [2020, 2021]},
        "variance": 0.2,
        "warnings": [],
        "provenance": ["fixture"],
    }


def test_slice16b_builds_selection_plan_from_candidate_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "pirs_field_candidates.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 3,
            "rejected_count": 1,
            "candidates": [
                _candidate("outcome_low", "outcome", 1.0),
                _candidate("outcome_high", "outcome", 10.0),
                _candidate("covariate_a", "covariate", 3.0),
            ],
            "rejected": [{"field_id": "metadata_only", "reason": "metadata_only_field_not_model_eligible"}],
        }),
        encoding="utf-8",
    )
    candidates, rejected, payload = load_pirs_candidate_manifest(manifest)
    assert len(candidates) == 3
    assert rejected[0]["field_id"] == "metadata_only"
    assert payload["gate"] == "pirs_candidate_gate"

    plan = build_pirs_selection_plan(candidate_manifest=manifest, budget="standard")
    assert plan.status == "planned"
    assert plan.selected_outcome_field_id == "outcome_high"
    assert plan.selected_covariate_field_ids == ("covariate_a",)
    assert plan.residual_mode == "cross_fitted"
    assert plan.fold_scheme["n_folds"] == 5
    assert len(plan.gate_rejected) == 1
    assert "pirs_candidate_gate_rejected_fields_before_selection" in plan.warnings


def test_slice16b_blocks_plan_without_outcome(tmp_path: Path) -> None:
    manifest = tmp_path / "pirs_field_candidates.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 1,
            "rejected_count": 0,
            "candidates": [_candidate("covariate_only", "covariate", 5.0)],
            "rejected": [],
        }),
        encoding="utf-8",
    )
    plan = build_pirs_selection_plan(candidate_manifest=manifest, budget="fast")
    assert plan.status == "blocked_no_outcome"
    assert plan.selected_outcome_field_id is None
    assert "pirs_selection_has_no_model_eligible_outcome" in plan.warnings
