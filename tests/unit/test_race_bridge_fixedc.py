import json
from pathlib import Path

import pytest

from pegasus.efg.race_bridge import (
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
