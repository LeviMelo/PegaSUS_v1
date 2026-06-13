from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from pegasus.cli import app
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic_report import inspect_hsic_report_manifest, run_export_hsic_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _seed_ranked_hsic_run(run_dir: Path) -> None:
    create_empty_output_bundle(run_dir)
    tables = run_dir / "Tables"
    cards = [
        {
            "rank": 1,
            "hypothesis_id": "hsic__resid__cov_a",
            "title": "HSIC residual signal: cov_a",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_a",
            "metrics": {"statistic": 0.9, "p_value": 0.01, "q_value": 0.04, "n_eff": 40.0},
            "state": "exploratory",
            "evidence_tier": "supported_descriptive",
            "decision": "retain_for_review",
            "warnings": [],
            "dashboard_safe": True,
            "read_only": True,
        },
        {
            "rank": 2,
            "hypothesis_id": "hsic__resid__cov_b",
            "title": "HSIC residual signal: cov_b",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_b",
            "metrics": {"statistic": 0.2, "p_value": 0.5, "q_value": 0.8, "n_eff": 8.0},
            "state": "fragile",
            "evidence_tier": "fragile_descriptive",
            "decision": "show_as_fragile",
            "warnings": ["hsic_descriptive_small_support"],
            "dashboard_safe": True,
            "read_only": True,
        },
    ]
    _write_json(tables / "dashboard_hsic_cards.json", {
        "schema_version": "1.0",
        "slice": "18B",
        "read_only": True,
        "card_count": len(cards),
        "cards": cards,
    })
    _write_json(tables / "hsic_residual_scan_ranking_manifest.json", {
        "schema_version": "1.0",
        "slice": "18B",
        "status": "ranked",
        "rank_count": len(cards),
        "dashboard_card_count": len(cards),
        "ranked_hypotheses": cards,
    })


def test_slice18c_exports_hsic_report_without_mutating_output_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_ranked_hsic_run(run_dir)
    payload = run_export_hsic_report(run_dir=run_dir)
    assert payload["status"] == "exported"
    assert payload["evidence_count"] == 2
    assert (run_dir / "Tables" / "hsic_evidence_report.json").exists()
    assert (run_dir / "Tables" / "hsic_evidence_report.md").exists()
    assert (run_dir / "Tables" / "hsic_evidence_report.csv").exists()
    assert (run_dir / "Tables" / "hsic_evidence_report_manifest.json").exists()
    text = (run_dir / "Tables" / "hsic_evidence_report.md").read_text(encoding="utf-8")
    assert "PegaSUS HSIC residual evidence report" in text
    assert "cov_a" in text
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["hsic_report_gate"]["status"] == "exported"
    inspect = inspect_hsic_report_manifest(run_dir / "Tables" / "hsic_evidence_report_manifest.json")
    assert inspect["read_only"] is True
    assert inspect["evidence_count"] == 2


def test_slice18c_cli_exports_and_inspects_hsic_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_ranked_hsic_run(run_dir)
    runner = CliRunner()
    result = runner.invoke(app, ["pirs", "export-hsic-report", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "hsic_report_gate" in result.output or "evidence_count" in result.output
    manifest = run_dir / "Tables" / "hsic_evidence_report_manifest.json"
    inspect_result = runner.invoke(app, ["pirs", "inspect-hsic-report", "--manifest", str(manifest)])
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "exported" in inspect_result.output
