
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run


def test_slice16c_attaches_design_gate_without_changing_first_class_bundle_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    selection_path = run_dir / "Tables" / "pirs_selection_plan.json"
    selection_path.write_text(
        json.dumps(
            {
                "budget": "standard",
                "selected_outcome_field_id": "field:outcome",
                "selected_covariate_field_ids": ["field:cov_a"],
                "selected_offset_field_id": None,
                "residual_mode": "cross_fitted",
                "family": "gaussian_identity",
                "fold_scheme": {"mode": "cross_fitted", "fold_count": 5},
                "gate_rejected": [{"field_id": "field:quarantined", "reason": "metadata_only"}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = run_attach_pirs_design_plan_to_run(run_dir=run_dir, selection_plan=selection_path)
    manifest_path = run_dir / "Tables" / "pirs_design_plan.json"

    assert payload["status"] == "planned"
    assert manifest_path.exists()
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        attached = json.loads((run_dir / rel).read_text(encoding="utf-8"))
        assert attached["pirs_design_gate"]["status"] == "planned"
        assert attached["pirs_design_gate"]["design_matrix_state"] == "planned_only"
        assert attached["pirs_design_gate"]["model_fit_state"] == "not_started"
    assert validate_output_bundle(run_dir=str(run_dir)).ok
