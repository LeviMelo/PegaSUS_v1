from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.dashboard.hsic_readonly import inspect_hsic_dashboard_cards
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic_rank import run_rank_hsic_residual_scan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        tables = run_dir / "Tables"
        rows = [
            {
                "hypothesis_id": "hsic__resid__cov_a",
                "residual_field_id": "resid",
                "covariate_field_id": "cov_a",
                "statistic": 1.2,
                "p_value": 0.01,
                "q_value": 0.04,
                "n_eff": 30.0,
                "state": "exploratory",
                "hsic_mode": "exact_linear",
                "residual_mode": "in_sample",
                "fold_scheme": "none",
                "warnings": [],
            }
        ]
        pq.write_table(pa.Table.from_pylist(rows), tables / "hsic_residual_scan_scores.parquet")
        _write_json(tables / "hsic_residual_scan_manifest.json", {
            "schema_version": "1.0",
            "slice": "18A",
            "status": "scanned",
            "score_count": 1,
            "hypothesis_count": 1,
            "scan_rows": rows,
        })
        payload = run_rank_hsic_residual_scan(run_dir=run_dir)
        if payload.get("status") != "ranked":
            errors.append("ranking status was not ranked")
        if payload.get("rank_count") != 1:
            errors.append("rank count mismatch")
        ranking_path = run_dir / "Tables" / "hsic_residual_scan_ranking.parquet"
        dashboard_path = run_dir / "Tables" / "dashboard_hsic_cards.json"
        if not ranking_path.exists():
            errors.append("ranking parquet missing")
        if not dashboard_path.exists():
            errors.append("dashboard cards json missing")
        cards = inspect_hsic_dashboard_cards(dashboard_path)
        if cards.get("read_only") is not True:
            errors.append("dashboard cards are not read-only")
        if cards.get("card_count") != 1:
            errors.append("dashboard card count mismatch")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.extend(validation.errors)
        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        if run_config.get("hsic_ranking_gate", {}).get("status") != "ranked":
            errors.append("run summary missing hsic_ranking_gate")
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 18B HSIC ranking and dashboard read-only exposure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
