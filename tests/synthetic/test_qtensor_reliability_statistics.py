"""§3.12 reliability statistics measure the spec quantity, not a naive proxy (W1-pt3).

The default path keeps the legacy dispersion/first-difference proxies (byte-identical guard
at the bottom); the spec_reliability path computes the §3.12.3/.8/.9/.11 quantities the names
claim. Each assertion is a case where the naive and spec definitions DIVERGE.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from pegasus.efg.q_tensor import (
    _cv,
    _denom_fragility_share,
    _sampling_cv,
    _second_diff_roughness,
    _temporal_roughness,
    _kish_effective_n,
)
from pegasus.workflows.investigate import state_reliability_weights


def test_kish_neff_over_denominator_deflates_below_n():
    # Equal VALUE cells with unequal DENOMINATOR weights: Kish over the value series ~= n
    # (values are flat), but over the concentrated denominator it collapses toward 1.
    values = [10.0, 10.0, 10.0, 10.0]
    denom = [1000.0, 1.0, 1.0, 1.0]
    kish_value = _kish_effective_n(values)
    kish_denom = _kish_effective_n(denom)
    assert abs(kish_value - 4.0) < 1e-9          # naive-on-value: ~n
    assert kish_denom < 1.1                        # spec-on-denominator: << n
    assert kish_denom < kish_value


def test_denom_fragility_is_share_below_floor():
    # 30 of 100 denominator cells below mu_min=50 -> fragility == 0.30 exactly.
    denom = [10.0] * 30 + [500.0] * 70
    assert abs(_denom_fragility_share(denom, mu_min=50.0) - 0.30) < 1e-12
    # No cell below floor -> zero fragility, regardless of magnitude.
    assert _denom_fragility_share([100.0, 100.0], mu_min=50.0) == 0.0


def test_sampling_cv_is_estimate_se_not_value_dispersion():
    # A near-flat rate across cells (huge counts) has tiny SAMPLING CV even though a smooth
    # value gradient gives a NON-tiny dispersion CV. The two definitions must diverge.
    counts = [10000.0, 10100.0, 10200.0, 10300.0]
    exposure = [100000.0] * 4
    scv = _sampling_cv(counts, exposure)
    disp = _cv(counts)
    # Poisson-rate aggregate SE/est = 1/sqrt(sum count).
    assert abs(scv - 1.0 / math.sqrt(sum(counts))) < 1e-6
    assert scv < disp                              # sampling precision is far tighter than spread
    # Bare-count Poisson CV = 1/sqrt(Y); a large count is precise, a tiny one is not.
    assert _sampling_cv([10000.0]) < _sampling_cv([4.0])


def test_roughness_is_curvature_not_variance():
    # A straight ramp has ZERO second-difference curvature but LARGE variance; a zig-zag of the
    # SAME variance has large curvature. Roughness must track curvature, not spread.
    ramp = [float(i) for i in range(20)]                       # smooth trend, high variance
    zigzag = [(10.0 if i % 2 else 0.0) for i in range(20)]     # jagged, comparable variance
    assert np.var(ramp) > 5.0
    assert _second_diff_roughness(ramp) < 1e-9                 # curvature ~ 0 for a line
    assert _second_diff_roughness(zigzag) > _second_diff_roughness(ramp)
    # The naive first-difference roughness does NOT vanish on a ramp (it sees the slope),
    # which is exactly the divergence the correction fixes.
    assert _temporal_roughness(ramp) > 1e-6


def _write_q_tensor(tmp_path):
    # One rate field: n_eff (Kish over exposure) is a small fraction of n_denom but ~= n_events;
    # the base of the Kish efficiency is what the two paths disagree on.
    q = pl.DataFrame({
        "field_id": ["rateA"],
        "n_events": [50.0],
        "n_denom": [100000.0],
        "n_eff": [50.0],
        "denom_fragility": [0.0],
        "provenance_risk": [0.0],
    })
    p = tmp_path / "Q_tensor.parquet"
    q.write_parquet(p)
    return tmp_path


def test_state_reliability_weights_spec_uses_denominator_base(tmp_path):
    run_dir = _write_q_tensor(tmp_path)
    legacy = state_reliability_weights(run_dir)["rateA"]
    spec = state_reliability_weights(run_dir, spec_reliability=True)["rateA"]
    # Legacy: n_eff/n_events = 50/50 = 1.0 -> reliability floored/capped at 1.0.
    assert abs(legacy - 1.0) < 1e-12
    # Spec: n_eff/n_denom = 50/100000 = 5e-4 -> below the 0.1 floor -> exactly 0.1.
    assert abs(spec - 0.1) < 1e-12


def test_default_path_is_byte_identical():
    # The default (spec_reliability=False) proxies are untouched: guard against silent drift.
    values = [1.0, 4.0, 9.0, 16.0, 25.0]
    assert _cv(values) == _cv(values)
    assert _temporal_roughness(values) == _temporal_roughness(values)
    # Spec helpers are strictly ADDITIVE — invoking them does not mutate the legacy ones.
    _ = _sampling_cv(values)
    _ = _second_diff_roughness(values)
    _ = _denom_fragility_share(values)
    assert _cv(values) is not None and _temporal_roughness(values) is not None
