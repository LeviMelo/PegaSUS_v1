
from __future__ import annotations

from pegasus.efg.promotion_apply import planned_promotion_fields, q_tensor_row, v_field_row, variable_dictionary_row


def _field() -> dict:
    return {
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


def test_slice15b_selects_only_promotable_metadata_only_fields() -> None:
    plan = {
        "promotions": [
            {"status": "planned", "action": "promote", "field": _field()},
            {"status": "blocked", "blocked": True, "field": {**_field(), "field_id": "blocked"}},
            {"status": "planned", "action": "defer", "field": {**_field(), "field_id": "deferred"}},
        ]
    }
    promoted, skipped, blocked = planned_promotion_fields(plan)
    assert [field["field_id"] for field in promoted] == ["efg_substrate__SIM_DO__underlying_icd_norm"]
    assert [item["field_id"] for item in skipped] == ["deferred"]
    assert [item["field_id"] for item in blocked] == ["blocked"]


def test_slice15b_builds_output_rows_for_validator_contract() -> None:
    field = _field()
    v = v_field_row(field)
    q = q_tensor_row(field)
    vd = variable_dictionary_row(field)
    assert v["field_id"] == field["field_id"]
    assert v["unit"] == "ICD10"
    assert v["state"] == "quarantined_descriptive"
    assert v["materialization_state"] == "metadata_only"
    assert q["field_id"] == field["field_id"]
    assert q["state"] == "quarantined_descriptive"
    assert vd["field_id"] == field["field_id"]
    assert vd["estimand_label"] == "metadata_only_observer_not_model_estimand"
