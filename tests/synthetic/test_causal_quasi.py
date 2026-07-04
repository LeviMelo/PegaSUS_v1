"""CAUSAL-02 — rung-2 quasi-experimental leverage (MSD-III §IV): ITS + DiD."""

from __future__ import annotations

import numpy as np

from pegasus.causal.quasi import (
    detect_structural_break,
    difference_in_differences,
    interrupted_time_series,
)


def test_its_recovers_level_jump() -> None:
    rng = np.random.default_rng(0)
    n, t0 = 60, 30
    t = np.arange(n)
    y = 2.0 + 0.1 * t + 5.0 * (t >= t0) + rng.normal(0, 0.5, n)   # planted +5 level jump

    res = interrupted_time_series(y, t0)
    assert abs(res.level_change - 5.0) < 1.0
    assert res.level_t > 3.0                                       # a significant interruption


def test_detect_break_near_shock() -> None:
    rng = np.random.default_rng(1)
    n, t0 = 60, 35
    y = 1.0 + 6.0 * (np.arange(n) >= t0) + rng.normal(0, 0.4, n)
    idx = detect_structural_break(y)
    assert idx is not None and abs(idx - t0) <= 2


def test_did_recovers_treatment_effect_net_of_trend() -> None:
    rng = np.random.default_rng(2)
    n = 40
    pre = np.arange(n) < 20
    post = ~pre
    treated = 10.0 + rng.normal(0, 0.3, n) + 2.0 * post + 3.0 * post   # common +2 trend, treatment +3
    control = 8.0 + rng.normal(0, 0.3, n) + 2.0 * post                 # only the common trend

    res = difference_in_differences(treated, control, pre_mask=pre, post_mask=post)
    assert abs(res.effect - 3.0) < 0.5                                 # trend netted out
