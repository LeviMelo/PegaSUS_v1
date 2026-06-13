
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run
from pegasus.efg.promotion_plan import write_efg_promotion_plan
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle


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


def test_slice15c_real_promotion_apply_chain_preserves_support_and_axes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    materialization = tmp_path / "efg_substrate_materialization.json"
    materialization.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "efg_substrate_materialization",
                "fields": [{"candidate_id": "candidate_underlying_icd_norm", "field": _field()}],
                "excluded_source_fields": [{"column": "constant_col", "efg_materialized": False}],
            }
        ),
        encoding="utf-8",
    )

    plan_result = write_efg_promotion_plan(run_dir=run_dir, materialization_manifest=materialization)
    apply_result = apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_result["path"])
    assert apply_result.promoted_field_count == 1
    assert validate_output_bundle(run_dir=str(run_dir)).ok

    vd = pq.read_table(run_dir / "VariableDictionary.parquet").to_pylist()
    row = next(item for item in vd if item["field_id"] == "efg_substrate__SIM_DO__underlying_icd_norm")
    assert "underlying_icd_norm" in row["support_description"]
    assert "icd10" in row["axis_description"]
