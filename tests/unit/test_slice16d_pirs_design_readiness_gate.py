
from __future__ import annotations

from pegasus.pirs.design_readiness import design_readiness_rejection_reasons


def test_slice16d_rejects_metadata_only_quarantined_fields() -> None:
    reasons = design_readiness_rejection_reasons(
        field={
            "field_id": "efg__x",
            "state": "quarantined_descriptive",
            "materialization_state": "metadata_only",
            "dashboard_safe": False,
        },
        q_state={
            "field_id": "efg__x",
            "state": "quarantined_descriptive",
            "warnings": ["q_tensor_placeholder_no_numerical_tensor"],
            "n_eff": 0,
            "missingness": 1.0,
        },
    )
    assert "field_state_not_model_ready:quarantined_descriptive" in reasons
    assert "materialization_state_not_tensor_backed:metadata_only" in reasons
    assert "field_not_dashboard_safe" in reasons
    assert "q_state_not_model_ready:quarantined_descriptive" in reasons
    assert "q_tensor_placeholder:q_tensor_placeholder_no_numerical_tensor" in reasons
    assert "q_tensor_n_eff_nonpositive" in reasons
    assert "q_tensor_missingness_complete" in reasons


def test_slice16d_accepts_tensor_backed_verified_field() -> None:
    reasons = design_readiness_rejection_reasons(
        field={
            "field_id": "sim__rate",
            "state": "verified",
            "materialization_state": "materialized",
            "dashboard_safe": True,
        },
        q_state={
            "field_id": "sim__rate",
            "state": "verified",
            "warnings": [],
            "n_eff": 100,
            "missingness": 0.1,
        },
    )
    assert reasons == ()
