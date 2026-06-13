
from __future__ import annotations

from pathlib import Path

from pegasus.pirs.design_plan import build_pirs_design_plan, pirs_design_plan_summary, write_pirs_design_plan


def _selection_payload() -> dict:
    return {
        "budget": "standard",
        "selected_outcome_field_id": "field:outcome",
        "selected_covariate_field_ids": ["field:cov_a", "field:cov_b"],
        "selected_offset_field_id": "field:population",
        "residual_mode": "cross_fitted",
        "family": "poisson_rate",
        "fold_scheme": {"mode": "cross_fitted", "fold_count": 5},
        "selection_rejected": [{"field_id": "field:bad", "reason": "zero_variance"}],
        "warnings": ["selection_warning"],
    }


def test_slice16c_builds_non_mutating_design_plan_from_selection_payload() -> None:
    plan = build_pirs_design_plan(_selection_payload())
    manifest = plan.as_manifest()

    assert manifest["artifact"] == "pirs_design_plan"
    assert manifest["status"] == "planned"
    assert manifest["design_matrix_state"] == "planned_only"
    assert manifest["model_fit_state"] == "not_started"
    assert manifest["residual_state"] == "not_started"
    assert manifest["hsic_state"] == "not_started"
    assert manifest["outcome_field_id"] == "field:outcome"
    assert manifest["covariate_field_ids"] == ["field:cov_a", "field:cov_b"]
    assert manifest["offset_field_id"] == "field:population"
    assert {term["role"] for term in manifest["terms"]} == {"intercept", "outcome", "covariate", "offset"}
    assert manifest["rejected"][0]["reason"] == "zero_variance"


def test_slice16c_blocks_design_plan_without_outcome_or_covariates() -> None:
    plan = build_pirs_design_plan({"budget": "fast", "selected_covariate_field_ids": []})
    manifest = plan.as_manifest()
    assert manifest["status"] == "blocked"
    assert "pirs_design_plan_missing_outcome" in manifest["warnings"]
    assert "pirs_design_plan_missing_covariates" in manifest["warnings"]


def test_slice16c_writes_design_plan_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out = run_dir / "Tables" / "pirs_design_plan.json"
    payload = write_pirs_design_plan(run_dir=run_dir, selection_plan=_selection_payload(), output=out)
    assert out.exists()
    summary = pirs_design_plan_summary(payload, manifest_path=out)
    assert summary["status"] == "planned"
    assert summary["covariate_count"] == 2
    assert summary["non_mutating"] is True
