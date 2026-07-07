"""O3 / §V.6(3) — exact-certifies-approximate wired live on a real spatial slice.

The national LDO uses a randomized low-rank readout. It must be CERTIFIED against an exact refit
on a bounded real slice: overlapping edges must agree within propagated bounds, else the
approximation is rejected LOUDLY. The comparator was orphaned (zero live callers); this pins the
live runner + its strict-rejection behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from pegasus.ldo.exact_certify import (
    ApproximationRejectedError,
    certify_exact_vs_approx,
)
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.records import LinkRecord
from pegasus.validation.holdout import certify_approximation_on_slice


def _field(p=60, S=200, T=8, seed=0):
    # p > 2·factor_rank_cap(24) → the randomized readout is used; a couple of real edges planted.
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((p, S, T))
    Z[1] += 0.7 * Z[0]
    Z[3] -= 0.6 * Z[2]
    return GaussianField(variables=tuple(f"v{i}" for i in range(p)),
                         space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((p, S, T)), resolution="year")


def test_certification_runs_and_agrees_on_a_real_slice():
    report = certify_approximation_on_slice(_field(), K=1, max_slice_S=120, seed=1)
    assert report["ran"] is True
    assert report["slice_S"] <= 120
    # the randomized and exact readouts of the same slice agree within propagated bounds
    assert report["certified"] is True
    assert report["n_disagree"] == 0


def test_strict_mode_raises_on_planted_disagreement():
    # two edge sets that share an endpoint/lag key but disagree far beyond a zero band
    approx = [LinkRecord(source_var="A", target_var="B", edge_type="contemporaneous",
                         weight=0.9, uncertainty=0.01)]
    exact = [LinkRecord(source_var="A", target_var="B", edge_type="contemporaneous",
                        weight=0.1, uncertainty=0.01)]
    rep = certify_exact_vs_approx(approx, exact, tolerance_sigma=3.0)
    assert rep["certified"] is False and rep["n_disagree"] == 1
    with pytest.raises(ApproximationRejectedError):
        certify_exact_vs_approx(approx, exact, tolerance_sigma=3.0, strict=True)


def test_run_ldo_records_the_certification_in_diagnostics():
    from pegasus.ldo.orchestrator import run_ldo
    run = run_ldo(_field(p=60, S=120, T=6), K=1, n_subsamples=2, run_residual_scan=False, seed=0)
    cert = run.diagnostics["exact_certifies_approx"]
    assert cert is not None and cert.get("ran") is True     # certification actually ran on the national-ish fit
