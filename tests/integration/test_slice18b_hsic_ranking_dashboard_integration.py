from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from pegasus.cli import app
from pegasus.dashboard.hsic_readonly import inspect_hsic_dashboard_cards
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic_rank import run_rank_hsic_residual_scan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _seed_scanned_run(run_dir: Path) -> None:
    create_empty_output_bundle(run_dir)
    tables = run_dir / "Tables"
    rows = [
        {
            "hypothesis_id": "hsic__resid__cov_a",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_a",
            "statistic": 0.7,
            "p_value": 0.03,
            "q_value": 0.06,
            "n_eff": 28.0,
            "state": "exploratory",
            "hsic_mode": "exact_linear",
            "residual_mode": "in_sample",
            "fold_scheme": "none",
            "warnings": [],
        },
        {
            "hypothesis_id": "hsic__resid__cov_b",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_b",
            "statistic": 0.1,
            "p_value": 0.7,
            "q_value": 0.9,
            "n_eff": 28.0,
            "state": "fragile",
            "hsic_mode": "exact_linear",
            "residual_mode": "in_sample",
            "fold_scheme": "none",
            "warnings": ["hsic_descriptive_small_support"],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tables / "hsic_residual_scan_scores.parquet")
    _write_json(tables / "hsic_residual_scan_manifest.json", {
        "schema_version": "1.0",
        "slice": "18A",
        "status": "scanned",
        "score_count": 2,
        "hypothesis_count": 2,
        "scan_rows": rows,
    })


def test_slice18b_ranks_hsic_scan_writes_dashboard_and_preserves_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_scanned_run(run_dir)
    payload = run_rank_hsic_residual_scan(run_dir=run_dir)
    assert payload["status"] == "ranked"
    assert payload["rank_count"] == 2
    assert payload["dashboard_card_count"] == 2
    assert (run_dir / "Tables" / "hsic_residual_scan_ranking.parquet").exists()
    assert (run_dir / "Tables" / "dashboard_hsic_cards.json").exists()
    assert validate_output_bundle(run_dir=str(run_dir)).ok
    cards = inspect_hsic_dashboard_cards(run_dir / "Tables" / "dashboard_hsic_cards.json")
    assert cards["read_only"] is True
    assert cards["card_count"] == 2
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["hsic_ranking_gate"]["status"] == "ranked"


def test_slice18b_cli_ranks_and_inspects_hsic_dashboard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_scanned_run(run_dir)
    runner = CliRunner()
    result = runner.invoke(app, ["pirs", "rank-hsic-scan", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "hsic_ranking_gate" in result.output or "ranked_hypotheses" in result.output
    manifest = run_dir / "Tables" / "hsic_residual_scan_ranking_manifest.json"
    dashboard = run_dir / "Tables" / "dashboard_hsic_cards.json"
    inspect_result = runner.invoke(app, ["pirs", "inspect-hsic-ranking", "--manifest", str(manifest)])
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "ranked" in inspect_result.output
    dash_result = runner.invoke(app, ["pirs", "inspect-hsic-dashboard", "--dashboard", str(dashboard)])
    assert dash_result.exit_code == 0, dash_result.output
    assert "read_only" in dash_result.output
