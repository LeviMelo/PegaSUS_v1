
from __future__ import annotations

from pathlib import Path

from pegasus.acceptance.contracts import assert_dashboard_did_not_compute, summarize_run
from pegasus.dashboard.read_only import inspect_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.acceptance import run_acceptance_check_run, run_acceptance_plan


def test_slice11a_acceptance_checks_valid_17_key_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    summary = summarize_run(run_dir)
    assert summary.ok, summary.errors
    assert summary.field_count == 1
    assert summary.q_count == 1
    assert "V_fields.parquet" in summary.first_class_keys
    assert "RunConfig.json" in summary.first_class_keys


def test_slice11a_acceptance_can_require_non_scaffold_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    result = run_acceptance_check_run(run_dir=run_dir, require_non_scaffold=True)
    assert result["ok"] is False
    assert any("non-scaffold" in error for error in result["errors"])


def test_slice11a_dashboard_inspection_does_not_change_acceptance_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    before = summarize_run(run_dir)
    inspected = inspect_run(run_dir=run_dir)
    after = summarize_run(run_dir)
    assert inspected["validation_ok"] is True
    assert_dashboard_did_not_compute(before=before, after=after)


def test_slice11a_workflow_plan_exposes_acceptance_surfaces() -> None:
    plan = run_acceptance_plan()
    assert "compile_smoke" in plan["surfaces"]
    assert "dashboard_read_only" in plan["surfaces"]
