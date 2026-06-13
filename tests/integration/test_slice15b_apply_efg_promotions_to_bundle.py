
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def test_slice15b_applies_promotion_plan_to_17_key_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    field = {
        "field_id": "efg_substrate__SIM_DO__underlying_icd_norm",
        "name": "SIM-DO underlying ICD observer",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "ICD10",
        "aggregation": "non_aggregable",
        "role": ["diagnostic_topology", "source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "underlying_icd_norm"},
        "axes": {"diagnostic": "ICD10"},
        "provenance": ["fixture"],
        "warnings": [],
        "lineage_hash": "abc",
        "registry_hash": "registry",
        "materialization_state": "metadata_only",
    }
    plan_path = run_dir / "Tables" / "efg_promotion_plan.json"
    _write_json(plan_path, {"schema_version": "1.0", "promotions": [{"status": "planned", "action": "promote", "field": field}]})

    result = apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
    assert result.promoted_field_count == 1
    assert result.output_validation_ok is True

    v_ids = {row["field_id"] for row in pq.read_table(run_dir / "V_fields.parquet").to_pylist()}
    q_ids = {row["field_id"] for row in pq.read_table(run_dir / "Q_tensor.parquet").to_pylist()}
    vd_ids = {row["field_id"] for row in pq.read_table(run_dir / "VariableDictionary.parquet").to_pylist()}
    assert field["field_id"] in v_ids
    assert field["field_id"] in q_ids
    assert field["field_id"] in vd_ids

    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["efg_promotion_apply_gate"]["promoted_field_count"] == 1
    assert validate_output_bundle(run_dir=str(run_dir)).ok


def test_slice15b_apply_is_idempotent_by_field_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    plan_path = run_dir / "Tables" / "efg_promotion_plan.json"
    field = {
        "field_id": "efg_substrate__SIM_DO__year",
        "name": "SIM-DO year observer",
        "kind": "observer_proxy",
        "carrier": "Deaths",
        "unit": "year",
        "aggregation": "non_aggregable",
        "role": ["source_field", "substrate_materialized"],
        "source": ["SIM-DO"],
        "support": {"column": "year"},
        "axes": {"time": "year"},
        "provenance": ["fixture"],
        "materialization_state": "metadata_only",
    }
    _write_json(plan_path, {"promotions": [{"status": "planned", "action": "promote", "field": field}]})
    apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
    apply_efg_promotion_plan_to_run(run_dir=run_dir, promotion_plan=plan_path)
    rows = pq.read_table(run_dir / "V_fields.parquet").to_pylist()
    assert sum(1 for row in rows if row["field_id"] == field["field_id"]) == 1
