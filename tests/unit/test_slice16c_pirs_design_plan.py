
from __future__ import annotations

from pathlib import Path

from pegasus.pirs.design_plan import build_pirs_design_plan, pirs_design_plan_summary, write_pirs_design_plan
from pegasus.pirs.spatial import select_spatial_effect_mode


def _selection_payload() -> dict:
    return {
        "budget": "standard",
        "selected_outcome_field_id": "field:outcome",
        "selected_covariate_field_ids": ["field:cov_a", "field:cov_b"],
        "selected_offset_field_id": "field:population",
        "residual_mode": "cross_fitted",
        "family": "poisson_rate",
        "moran_i": 0.20,
        "spatial_missingness": 0.05,
        "time_period_count": 12,
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
    assert manifest["spatial_effect_mode"] == "municipality_FE"
    assert manifest["spatial_effect"]["reason"] == "long_panel_low_spatial_missingness_selects_municipality_fixed_effects"
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


def test_slice16c_spatial_selector_precedence() -> None:
    fast = select_spatial_effect_mode(
        budget="fast",
        time_period_count=20,
        spatial_missingness=0.0,
        moran_i=0.0,
    )
    assert fast.mode == "UF_FE"

    none = select_spatial_effect_mode(
        budget="standard",
        time_period_count=20,
        spatial_missingness=0.0,
        moran_i=0.0,
    )
    assert none.mode == "none"

    icar = select_spatial_effect_mode(
        budget="standard",
        time_period_count=4,
        spatial_missingness=0.20,
        moran_i=0.30,
    )
    assert icar.mode == "ICAR"
    # ICAR is implemented as a penalized-IRLS GMRF (model_execution._icar_design /
    # _crossfit_icar_residuals), so the selector no longer gates to descriptive-only;
    # it requires a declared municipality adjacency artifact to execute (PIRS-SPAT-01,
    # plan option (a) was implemented rather than the option (b) descriptive fallback).
    assert "icar_requires_declared_municipality_adjacency" in icar.warnings
