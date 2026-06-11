from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic import run_hsic_plan_fixture, run_hsic_build_fixture

FIXTURE = Path("tests/fixtures/pirs/pirs_slice9a_hsic_fixture.json")


def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def test_slice9a_hsic_plan_selects_cross_fitted_exact_mode() -> None:
    plan = run_hsic_plan_fixture(input_path=FIXTURE, budget="standard")
    assert plan["hsic_mode"] == "exact"
    assert plan["residual_mode"] == "cross_fitted"
    assert plan["null_strategy"] == "spatial_block_cyclic_time_shift"
    assert plan["fdr_method"] == "BY"


def test_slice9a_hsic_bundle_validates_and_preserves_contracts(tmp_path: Path) -> None:
    run_dir = tmp_path / "slice9a"
    run_hsic_build_fixture(input_path=FIXTURE, run_dir=run_dir, budget="standard")
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
    ids = {row["field_id"] for row in v}
    assert "hsic_outcome_residual_deviance" in ids
    assert "hsic_covariate_context" in ids
    assert "hsic_residual_association_score" in ids
    by_id = {row["field_id"]: row for row in v}
    residual_prov = _loads(by_id["hsic_outcome_residual_deviance"]["provenance"])
    assert residual_prov == ["model_derived"]
    hsic_support = _loads(by_id["hsic_residual_association_score"]["support_json"])
    metadata = hsic_support["field_metadata"]
    assert metadata["hsic_mode"] == "exact"
    assert metadata["residual_mode"] == "cross_fitted"
    assert metadata["null_strategy"] == "spatial_block_cyclic_time_shift"
    assert by_id["hsic_residual_association_score"]["dashboard_safe"] == "False"

    h = pl.read_parquet(run_dir / "Hypotheses.parquet").to_dicts()
    assert h and h[0]["residual_field_id"] == "hsic_outcome_residual_deviance"
    assert h[0]["q_value"] is not None

    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text())
    assert manifest["pirs_hsic"]["approximation_diagnostics_emitted"] is True
    assert manifest["pirs_hsic"]["standard_deep_cross_fitted_residuals"] is True
    assert manifest["telemetry"]["stage_status"]["pirs_hsic"] == "success"
    assert (run_dir / "Tables" / "hsic_outputs.parquet").exists()
    assert (run_dir / "Tables" / "hsic_approximation_diagnostics.parquet").exists()
    assert (run_dir / "Tables" / "hsic_null_regime.parquet").exists()
    assert (run_dir / "Tables" / "hsic_fdr_correction.parquet").exists()


def test_slice9a_cuda_required_aborts_without_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "slice9a_cuda"
    run_hsic_build_fixture(input_path=FIXTURE, run_dir=run_dir, budget="standard", cuda_required=True)
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text())
    assert manifest["pirs_hsic"]["hsic_mode"] == "cuda_unavailable_abort"
    assert manifest["telemetry"]["stage_status"]["pirs_hsic"] == "blocked"
    failed = pl.read_parquet(run_dir / "FailedBranches.parquet").to_dicts()
    assert any(row["failed_branch_id"] == "slice9a_cuda_required_unavailable" for row in failed)
