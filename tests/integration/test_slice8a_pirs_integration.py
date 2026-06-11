from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.run_pirs import run_pirs_build_fixture, run_pirs_plan_fixture

FIXTURE = Path("tests/fixtures/pirs/pirs_slice8a_fixture.json")


def _loads(value):
    if value is None:
        return None
    return json.loads(value) if isinstance(value, str) else value


def test_slice8a_pirs_plan_reports_selection_and_residual_policy() -> None:
    plan = run_pirs_plan_fixture(input_path=FIXTURE, budget="standard")
    assert plan["status"] == "planned"
    assert plan["outcome_field_id"] == "pirs_fixture_deaths_outcome"
    assert plan["covariate_field_ids"] == ["pirs_fixture_cnes_capacity_covariate"]
    assert plan["offset_field_id"] == "pirs_fixture_population_offset"
    assert plan["family"] == "poisson_count_with_log_offset"
    assert plan["residual_mode"] == "cross_fitted"
    rejected = {r["field_id"]: r["reason"] for r in plan["rejected"]}
    assert rejected["pirs_fixture_zero_variance_covariate"] == "zero_variance_field_excluded_from_design_matrix"


def test_slice8a_pirs_bundle_validates_and_emits_model_residual_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "slice8a"
    result = run_pirs_build_fixture(input_path=FIXTURE, run_dir=run_dir, budget="standard")
    assert result["validation"].ok, result["validation"].errors
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors
    v = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
    by_id = {row["field_id"]: row for row in v}
    residual = by_id["pirs_residual_all_deaths"]
    assert residual["kind"] == "model_residual"
    assert _loads(residual["provenance"]) == ["model_derived"]
    assert residual["dashboard_safe"] == "False"
    model_assoc = pl.read_parquet(run_dir / "ModelAssociations.parquet").to_dicts()
    residual_assoc = pl.read_parquet(run_dir / "ResidualAssociations.parquet").to_dicts()
    assert model_assoc[0]["id"] == "pirs_model_poisson_all_deaths"
    assert model_assoc[0]["status"] == "fitted"
    assert residual_assoc[0]["id"] == "pirs_residual_all_deaths"
    detail = pl.read_parquet(run_dir / "Tables" / "pirs_model_associations_detail.parquet").to_dicts()[0]
    assert detail["offset_field_id"] == "pirs_fixture_population_offset"
    assert detail["family"] == "poisson_count_with_log_offset"
    diagnostics = pl.read_parquet(run_dir / "Tables" / "pirs_diagnostics.parquet").to_dicts()[0]
    assert diagnostics["residual_mode"] == "cross_fitted"
    assert diagnostics["zero_variance_rejections"] == 1
    failed = pl.read_parquet(run_dir / "FailedBranches.parquet").to_dicts()
    reasons = {row["reason"] for row in failed}
    assert "zero_variance_field_excluded_from_design_matrix" in reasons
    assert "q_state_illegal_excluded_not_model_eligible" in reasons
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert manifest["pirs"]["residual_mode"] == "cross_fitted"
    assert manifest["telemetry"]["stage_status"]["pirs_model"] == "success"
    assert manifest["telemetry"]["stage_status"]["pirs_hsic"] == "blocked"
