import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.sidra_context import run_sidra_context_build_fixture, run_sidra_context_plan_fixture


FIXTURE = Path("tests/fixtures/sidra/sidra_slice7_context_fixture.json")


def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _field_metadata(row):
    support = _loads(row.get("support_json"))
    if isinstance(support, dict) and isinstance(support.get("field_metadata"), dict):
        return support["field_metadata"]
    return _loads(row.get("metadata_json") or row.get("metadata"))


def test_slice7a_plan_reports_stitch_projection_highdim_and_stdfm() -> None:
    result = run_sidra_context_plan_fixture(input_path=FIXTURE)
    assert result["segments"] == 2
    assert result["stitch_status"] == "stitched"
    assert result["projection_status"] == "projected"
    assert "sidra_fractional_classification_projection" in result["projection_warnings"]
    assert result["high_dimensional_status"] == "bounded"
    assert result["stdfm_status"] == "blocked_solver_pending"


def test_slice7a_bundle_validates_and_preserves_sidra_stdfm_contracts(tmp_path: Path) -> None:
    run_dir = tmp_path / "slice7a"
    run_sidra_context_build_fixture(input_path=FIXTURE, run_dir=run_dir)
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
    ids = {row["field_id"] for row in v}
    assert "sidra_stitched_gdp_context" in ids
    assert "sidra_projected_labor_context" in ids
    assert "sidra_highdim_bounded_context" in ids
    assert "sidra_stdfm_blocked_candidate" in ids

    by_id = {row["field_id"]: row for row in v}
    projected_meta = _field_metadata(by_id["sidra_projected_labor_context"])
    assert projected_meta["projection_matrix_id"] == "slice7_cnae_to_sector_fractional_v1"
    assert projected_meta["projection"]["fractional"] is True

    highdim_meta = _field_metadata(by_id["sidra_highdim_bounded_context"])
    assert highdim_meta["high_dimensional_bound"]["status"] == "bounded"
    assert "occupation" in highdim_meta["high_dimensional_bound"]["axes_dropped"]

    stdfm = by_id["sidra_stdfm_blocked_candidate"]
    assert stdfm["state"] == "blocked"
    assert str(stdfm["dashboard_safe"]) == "False"
    stdfm_meta = _field_metadata(stdfm)
    assert stdfm_meta["stdfm_output"]["status"] == "blocked_solver_pending"

    warnings = pl.read_parquet(run_dir / "Warnings.parquet").to_dicts()
    codes = {row["code"] for row in warnings}
    assert "sidra_stitch_segment_provenance" in codes
    assert "sidra_fractional_classification_projection" in codes
    assert "high_dimensional_bounded_pushforward" in codes
    assert "blocked_solver_pending" in codes

    failed = pl.read_parquet(run_dir / "FailedBranches.parquet").to_dicts()
    reasons = "\n".join(row["reason"] for row in failed)
    assert "bounded pushforward" in reasons
    assert "blocked_solver_pending" in reasons

    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert manifest["sidra_context"]["stdfm_output"]["status"] == "blocked_solver_pending"
    assert (run_dir / "Tables" / "sidra_stitching_segments.parquet").exists()
    assert (run_dir / "Tables" / "sidra_projection_matrix.parquet").exists()
    assert (run_dir / "Tables" / "sidra_high_dimensional_bounds.parquet").exists()
    assert (run_dir / "Tables" / "stdfm_certification.parquet").exists()
