import json
from pathlib import Path

import pytest

from pegasus.measurement.race import (
    RaceBridgeCounts,
    RaceBridgeValidationError,
    fixedc_dynamic_weight_bridge,
    load_race_bridge_prior,
)


def test_valid_fixedc_prior_bridges_counts_and_preserves_missing():
    prior = load_race_bridge_prior("config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json")
    counts = RaceBridgeCounts(
        raw_admin_counts={"1": 2, "2": 1, "3": 0, "4": 3, "5": 0},
        missing_count=1,
        total_count=7,
        support={"time": {"years": [2022]}, "geography": {"municipality_cod6": ["270430"]}, "n_events": 7},
    )
    posterior = fixedc_dynamic_weight_bridge(counts, prior)
    assert posterior.missing_count == 1
    assert posterior.missing_share == pytest.approx(1 / 7)
    assert posterior.raw_admin_counts["4"] == 3
    assert posterior.posterior_counts["parda"] > posterior.posterior_counts["preta"]
    assert posterior.sensitivity_width >= posterior.missing_share
    assert posterior.metadata()["bridge_mode"] == "fixedC_dynamic_weight"


def test_empty_prior_blocks_posterior_output():
    with pytest.raises(RaceBridgeValidationError):
        load_race_bridge_prior("tests/fixtures/race_bridge/fixedC_invalid_empty.json")


def test_prior_rows_must_sum_to_one():
    with pytest.raises(RaceBridgeValidationError):
        load_race_bridge_prior("tests/fixtures/race_bridge/fixedC_invalid_rowsum.json")


def test_race_bridge_cv_is_bootstrap_uncertainty_not_category_spread():
    prior = load_race_bridge_prior("config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json")
    counts = RaceBridgeCounts(
        raw_admin_counts={"1": 10, "2": 0, "3": 0, "4": 0, "5": 0},
        missing_count=0,
        total_count=10,
        support={"n_events": 10},
    )
    posterior = fixedc_dynamic_weight_bridge(counts, prior)
    values = list(posterior.posterior_counts.values())
    mean = sum(values) / len(values)
    spread_cv = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5 / mean
    assert posterior.race_bridge_cv != pytest.approx(spread_cv)
    assert posterior.race_bridge_cv > 0


def test_race_bridge_uses_local_population_shares_when_declared():
    prior = load_race_bridge_prior("config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json")
    counts = RaceBridgeCounts(
        raw_admin_counts={"1": 5, "2": 5, "3": 0, "4": 0, "5": 0},
        missing_count=0,
        total_count=10,
        support={
            "n_events": 10,
            "target_population_shares": {
                "branca": 0.1,
                "preta": 0.1,
                "amarela": 0.1,
                "parda": 0.6,
                "indigena": 0.1,
            },
        },
    )
    posterior = fixedc_dynamic_weight_bridge(counts, prior)
    # The §4.5.3 Bayes crosswalk W_{j,i} = C_{i,j}·π_i / Σ_m C_{m,j}·π_m does NOT
    # reduce to a naive total·π redistribution unless the emission matrix C is
    # uniform (it is not, here). With raw admin {1:5, 2:5} and a parda-heavy local
    # π (parda=0.6), the faithful posterior is
    #   5·W[1→parda] + 5·W[2→parda] = 5·(0.02·0.6/0.110) + 5·(0.10·0.6/0.150)
    #                                = 5·0.10909 + 5·0.40 = 2.5454...
    # A parda-heavy local prior still pulls mass toward parda relative to a uniform
    # prior, but the administrative signal (via C) is not overwritten — exactly the
    # epistemic guarantee §4 exists to enforce.
    assert posterior.effective_bridge_mode == "localPi_posteriorC"
    assert posterior.posterior_counts["parda"] == pytest.approx(2.5454545454545454)
