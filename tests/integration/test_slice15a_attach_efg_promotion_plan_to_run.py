
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.efg_promotion import run_attach_efg_promotion_plan_to_run


def test_slice15a_attaches_efg_promotion_plan_without_mutating_first_class_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    materialization_path = run_dir / "Tables" / "efg_substrate_materialization.json"
    materialization_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "materialized_fields": [
                    {
                        "field_id": "efg_substrate::underlying_icd_norm",
                        "name": "SIM-DO.underlying_icd_norm",
                        "kind": "observer_proxy",
                        "carrier": "Deaths",
                        "unit": "ICD10",
                        "dashboard_safe": "warning",
                        "materialization_state": "metadata_only",
                        "support": {"column": "underlying_icd_norm"},
                    }
                ],
                "excluded_source_fields": [
                    {"column": "constant_col", "reason": "zero_variance", "efg_materialized": False}
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    before = {p.name for p in run_dir.iterdir()}
    summary = run_attach_efg_promotion_plan_to_run(run_dir=run_dir)
    after = {p.name for p in run_dir.iterdir()}

    assert before == after
    assert summary["status"] == "planned"
    assert summary["planned_count"] == 1
    assert summary["conflict_count"] == 0
    assert summary["writes_v_fields"] is False
    assert (run_dir / "Tables" / "efg_promotion_plan.json").exists()

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["efg_promotion_gate"]["planned_count"] == 1
    assert validate_output_bundle(run_dir=str(run_dir)).ok
