from __future__ import annotations

from pegasus.pirs.hsic_ranking import rank_hsic_rows


def test_slice18b_ranks_by_state_q_value_and_statistic() -> None:
    rows = [
        {"hypothesis_id": "h_fragile", "state": "fragile", "q_value": 0.02, "p_value": 0.01, "statistic": 1.0, "n_eff": 4, "covariate_field_id": "b"},
        {"hypothesis_id": "h_verified", "state": "verified", "q_value": 0.50, "p_value": 0.20, "statistic": 0.1, "n_eff": 30, "covariate_field_id": "a"},
        {"hypothesis_id": "h_supported", "state": "exploratory", "q_value": 0.05, "p_value": 0.01, "statistic": 3.0, "n_eff": 40, "covariate_field_id": "c"},
    ]
    ranked = rank_hsic_rows(rows)
    assert [card.hypothesis_id for card in ranked] == ["h_verified", "h_fragile", "h_supported"]
    assert ranked[0].evidence_tier == "verified"
    assert ranked[1].evidence_tier == "fragile_descriptive"
    assert ranked[2].evidence_tier == "supported_descriptive"
    assert all(card.dashboard_safe for card in ranked)


def test_slice18b_merges_hypothesis_schema_values_when_scores_are_thin() -> None:
    rows = [{"hypothesis_id": "h1", "statistic": 2.0, "q_value": 0.2, "state": "exploratory"}]
    hypotheses = [{"hypothesis_id": "h1", "covariate_field_id": "cov_a", "residual_field_id": "resid", "n_eff": 11.0}]
    ranked = rank_hsic_rows(rows, hypotheses)
    assert ranked[0].covariate_field_id == "cov_a"
    assert ranked[0].residual_field_id == "resid"
    assert ranked[0].n_eff == 11.0
