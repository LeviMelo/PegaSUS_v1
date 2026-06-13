
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.pirs.selection_plan import attach_pirs_selection_plan_to_run, write_pirs_selection_plan


def _candidate(field_id: str, role: str, utility: float, carrier: str = "Deaths") -> dict:
    return {
        "field_id": field_id,
        "role": role,
        "utility": utility,
        "q_state": "verified",
        "carrier": carrier,
        "unit": "counts",
        "support": {"years": [2020], "municipalities": ["270430"]},
        "variance": 0.5,
        "warnings": [],
        "provenance": ["fixture"],
    }


def test_slice16b_attaches_selection_plan_to_run_without_mutating_first_class_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    candidate_manifest = run_dir / "Tables" / "pirs_field_candidates.json"
    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    candidate_manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 2,
            "rejected_count": 1,
            "candidates": [
                _candidate("verified_outcome", "outcome", 8.0),
                _candidate("verified_covariate", "covariate", 4.0),
            ],
            "rejected": [{"field_id": "efg_metadata_only", "reason": "metadata_only_field_not_model_eligible"}],
        }),
        encoding="utf-8",
    )

    manifest = write_pirs_selection_plan(run_dir=run_dir, candidate_manifest=candidate_manifest, budget="standard")
    assert manifest["status"] == "planned"
    assert manifest["selected_outcome_field_id"] == "verified_outcome"
    assert (run_dir / "Tables" / "pirs_selection_plan.json").exists()

    summary = attach_pirs_selection_plan_to_run(run_dir=run_dir, candidate_manifest=candidate_manifest, budget="standard")
    assert summary["status"] == "planned"
    assert summary["selected_covariate_count"] == 1
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["pirs_selection_gate"]["selected_outcome_field_id"] == "verified_outcome"
    assert p_vector["pirs_selection_gate"]["model_fitted"] is False
    assert p_vector["pirs_selection_gate"]["design_matrix_materialized"] is False
