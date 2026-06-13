
from __future__ import annotations

from pegasus.pirs.run_candidates import pirs_candidate_from_rows, pirs_candidate_rejection_reason


def _field(**updates):
    payload = {
        "field_id": "field_verified",
        "name": "Verified outcome",
        "kind": "extensive_measure",
        "carrier": "Deaths",
        "unit": "counts",
        "state": "verified",
        "dashboard_safe": "True",
        "materialization_state": "materialized",
        "support_json": '{"years":[2020]}',
        "warnings": "[]",
        "provenance": '["fixture"]',
    }
    payload.update(updates)
    return payload


def _q(**updates):
    payload = {
        "field_id": "field_verified",
        "n_eff": 50.0,
        "missingness": 0.0,
        "provenance_risk": 0.1,
        "cv": 0.2,
        "state": "verified",
        "warnings": "[]",
    }
    payload.update(updates)
    return payload


def test_slice16a_rejects_metadata_only_and_quarantined_promotions() -> None:
    reason = pirs_candidate_rejection_reason(
        _field(materialization_state="metadata_only", state="quarantined_descriptive", dashboard_safe="False"),
        _q(state="quarantined_descriptive", warnings='["q_tensor_placeholder_no_numerical_tensor"]'),
    )
    assert reason == "metadata_only_field_not_model_eligible"


def test_slice16a_rejects_placeholder_q_tensor_even_when_field_state_is_relaxed() -> None:
    reason = pirs_candidate_rejection_reason(
        _field(),
        _q(warnings='["q_tensor_placeholder_no_numerical_tensor"]'),
    )
    assert reason == "placeholder_q_tensor_not_model_eligible"


def test_slice16a_builds_field_candidate_for_verified_numeric_field() -> None:
    field = _field()
    q = _q()
    assert pirs_candidate_rejection_reason(field, q) is None
    candidate = pirs_candidate_from_rows(field, q)
    assert candidate.field_id == "field_verified"
    assert candidate.role == "outcome"
    assert candidate.q_state == "verified"
    assert candidate.utility > 0.0
