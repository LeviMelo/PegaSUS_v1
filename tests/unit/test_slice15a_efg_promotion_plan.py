
from __future__ import annotations

import json
from pathlib import Path

from pegasus.efg.promotion_plan import build_efg_promotion_plan, materialized_fields_from_manifest


def _manifest() -> dict:
    return {
        "schema_version": "1.0",
        "materialized_fields": [
            {
                "field_id": "efg_substrate::age_years",
                "name": "SIM-DO.age_years",
                "kind": "extensive_measure",
                "carrier": "Deaths",
                "unit": "years",
                "dashboard_safe": "true",
                "materialization_state": "metadata_only",
                "support": {"column": "age_years"},
            },
            {
                "field_id": "efg_substrate::underlying_icd_norm",
                "name": "SIM-DO.underlying_icd_norm",
                "kind": "observer_proxy",
                "carrier": "Deaths",
                "unit": "ICD10",
                "dashboard_safe": "warning",
                "materialization_state": "metadata_only",
                "support": {"column": "underlying_icd_norm"},
            },
        ],
        "excluded_source_fields": [
            {"column": "constant_col", "reason": "zero_variance", "efg_materialized": False}
        ],
    }


def test_slice15a_extracts_materialized_fields_from_manifest() -> None:
    fields = materialized_fields_from_manifest(_manifest())
    assert len(fields) == 2
    assert fields[1]["unit"] == "ICD10"


def test_slice15a_builds_non_mutating_promotion_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "Tables").mkdir(parents=True)
    manifest_path = run_dir / "Tables" / "efg_substrate_materialization.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    plan = build_efg_promotion_plan(run_dir=run_dir, materialization_manifest=manifest_path)

    assert plan.planned_count == 2
    assert plan.conflict_count == 0
    assert plan.excluded_source_field_count == 1
    assert plan.promotion_safe is True
    payload = plan.as_manifest()
    assert payload["contract"]["non_mutating"] is True
    assert payload["contract"]["writes_v_fields"] is False
    assert payload["planned_promotions"][1]["kind"] == "observer_proxy"
    assert payload["excluded_source_fields"][0]["efg_materialized"] is False
