
from __future__ import annotations

import json
from pathlib import Path

from pegasus.efg.promotion_apply import planned_promotion_fields
from pegasus.efg.promotion_plan import build_efg_promotion_plan, materialized_fields_from_manifest, write_efg_promotion_plan
from pegasus.output.bundle import create_empty_output_bundle


def _field() -> dict:
    return {
        "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
        "name": "SIM-DO.underlying_icd_norm",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "ICD10",
        "aggregation": "non_aggregable",
        "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "underlying_icd_norm", "row_count": 3, "source_system": "SIM-DO"},
        "axes": {"diagnosis": "icd10"},
        "operator": "metadata_only_substrate_materialization",
        "provenance": ["registry", "fixture"],
        "warnings": [],
        "lineage_hash": "lineage-demo",
        "registry_hash": "registry-demo",
        "materialization_state": "metadata_only",
        "dashboard_safe": "False",
    }


def test_slice15c_materialized_fields_accepts_slice14b_nested_shape() -> None:
    payload = {
        "fields": [
            {
                "candidate_id": "candidate_underlying_icd_norm",
                "field": _field(),
                "lineage_hash": "wrapper-lineage",
                "materialization_reason": "metadata_only",
            }
        ]
    }
    fields = materialized_fields_from_manifest(payload)
    assert len(fields) == 1
    assert fields[0]["field_id"] == "efg_substrate__SIM_DO__underlying_icd_norm"
    assert fields[0]["support"]["column"] == "underlying_icd_norm"
    assert fields[0]["candidate_id"] == "candidate_underlying_icd_norm"


def test_slice15c_promotion_plan_preserves_full_field_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    manifest = tmp_path / "efg_substrate_materialization.json"
    manifest.write_text(
        json.dumps({"fields": [{"candidate_id": "candidate_underlying_icd_norm", "field": _field()}]}),
        encoding="utf-8",
    )

    result = write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=manifest)
    planned = result["plan"]["planned_promotions"]
    assert len(planned) == 1
    assert planned[0]["field"]["support"]["column"] == "underlying_icd_norm"
    assert planned[0]["field"]["role"] == ["diagnostic_topology", "source_field", "substrate_materialized"]

    promoted, skipped, blocked = planned_promotion_fields(result["plan"])
    assert not skipped
    assert not blocked
    assert promoted[0]["support"]["column"] == "underlying_icd_norm"
    assert promoted[0]["axes"]["diagnosis"] == "icd10"
