from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic_report import inspect_hsic_report_manifest, run_export_hsic_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        tables = run_dir / "Tables"
        card = {
            "rank": 1,
            "hypothesis_id": "hsic__resid__cov_a",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_a",
            "metrics": {"statistic": 1.1, "p_value": 0.02, "q_value": 0.05, "n_eff": 30.0},
            "state": "exploratory",
            "evidence_tier": "supported_descriptive",
            "decision": "retain_for_review",
            "warnings": [],
            "dashboard_safe": True,
            "read_only": True,
        }
        _write_json(tables / "dashboard_hsic_cards.json", {
            "schema_version": "1.0",
            "slice": "18B",
            "read_only": True,
            "cards": [card],
        })
        _write_json(tables / "hsic_residual_scan_ranking_manifest.json", {
            "schema_version": "1.0",
            "slice": "18B",
            "status": "ranked",
            "rank_count": 1,
            "dashboard_card_count": 1,
            "ranked_hypotheses": [card],
        })
        payload = run_export_hsic_report(run_dir=run_dir)
        if payload.get("status") != "exported":
            errors.append("report status was not exported")
        if payload.get("evidence_count") != 1:
            errors.append("report evidence count mismatch")
        for name in ["hsic_evidence_report.json", "hsic_evidence_report.md", "hsic_evidence_report.csv", "hsic_evidence_report_manifest.json"]:
            if not (tables / name).exists():
                errors.append(f"missing report artifact: {name}")
        md = (tables / "hsic_evidence_report.md").read_text(encoding="utf-8")
        if "read-only descriptive export" not in md:
            errors.append("markdown report did not declare read-only export")
        inspect = inspect_hsic_report_manifest(tables / "hsic_evidence_report_manifest.json")
        if inspect.get("read_only") is not True:
            errors.append("inspect manifest did not preserve read_only true")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.extend(validation.errors)
        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        if run_config.get("hsic_report_gate", {}).get("status") != "exported":
            errors.append("run summary missing hsic_report_gate")
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 18C HSIC report export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
