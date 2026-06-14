from __future__ import annotations

import random

from pegasus.pirs.hsic import run_hsic_scan


def test_release_hsic_signal_ranks_above_deterministic_null() -> None:
    x = [float(index) / 30.0 for index in range(120)]
    signal = [value * value + 0.01 * (index % 3) for index, value in enumerate(x)]
    null = list(x)
    random.Random(91).shuffle(null)
    common = {
        "outcome_residual_field_id": "release_residual",
        "residuals": signal,
        "support_intersection": {"n_eff": 120},
        "budget": "standard",
        "null_strategy": "unrestricted_permutation",
        "fdr_method": "BY",
        "permutations": 39,
        "seed": 91,
    }
    signal_result = run_hsic_scan(covariate_field_id="signal", covariate=x, **common)
    null_result = run_hsic_scan(covariate_field_id="null", covariate=null, **common)
    repeat = run_hsic_scan(covariate_field_id="signal", covariate=x, **common)
    assert signal_result.statistic is not None and null_result.statistic is not None
    assert signal_result.statistic > null_result.statistic
    assert signal_result.p_value is not None and null_result.p_value is not None
    assert signal_result.p_value <= null_result.p_value
    assert signal_result.statistic == repeat.statistic
    assert signal_result.p_value == repeat.p_value
