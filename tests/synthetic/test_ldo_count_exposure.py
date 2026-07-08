"""WP3 / §III.5 — the count-with-exposure (Poisson-offset) margin, wired end to end.

The denominator principle (§I.2): an extensive count is modelled WITH its exposure as an
offset (dependence net of exposure), never as a pre-divided rate treated as Gaussian. The
margin (`count_exposure_gaussianize`) existed; this pins that a declared exposure tensor now
flows through `assemble → gaussianize → run_ldo` so the margin is actually applied.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.margins import count_exposure_gaussianize, gaussianize_field
from pegasus.ldo.orchestrator import run_ldo


def _count_field_with_exposure(seed: int = 0) -> LDOField:
    rng = np.random.default_rng(seed)
    p, S, T = 3, 40, 8
    exposure = np.full((p, S, T), np.nan)
    X = np.full((p, S, T), np.nan)
    # variable 0 is an extensive COUNT with a known population exposure (varies 10×);
    # a constant underlying rate → the count tracks exposure, which the margin nets out.
    pop = rng.uniform(1_000, 10_000, size=(S, T))
    exposure[0] = pop
    X[0] = rng.poisson(0.002 * pop).astype(float)  # count ≈ rate·exposure
    X[1] = rng.standard_normal((S, T))              # a plain continuous variable
    X[2] = rng.standard_normal((S, T)) + 0.5 * X[1]
    W = np.where(np.isfinite(X), 1.0, 0.0)
    return LDOField(variables=("Deaths", "B", "C"), space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                    time_ids=tuple(range(T)), X=X, W=W, resolution="year", exposure=exposure)


def test_exposure_drives_count_with_exposure_margin():
    field = _count_field_with_exposure()
    gf = gaussianize_field(field, seed=1, exposure=field.exposure)
    # variable 0 (count + exposure) must match the count-with-exposure margin exactly, and NOT the
    # plain rank margin — dependence on exposure is netted out in the latent Z. The margin uses the
    # per-municipality baseline (LDO-MARGIN-10), so the expectation is over the 2-D (S,T) field (the
    # same call gaussianize_field makes), not a flattened pooled-λ path.
    rng = np.random.default_rng(1)
    expect0 = count_exposure_gaussianize(field.X[0], field.exposure[0], rng=rng)
    assert np.allclose(np.nan_to_num(gf.Z[0]), np.nan_to_num(expect0), atol=1e-9)


def test_run_ldo_threads_exposure_and_reports_it():
    field = _count_field_with_exposure()
    run = run_ldo(field, K=2, n_subsamples=4, run_residual_scan=False, seed=0)
    # the LDOField carried an exposure tensor for one variable → the diagnostic reflects it
    assert run.diagnostics["count_exposure_variables"] == 1


def test_no_exposure_is_a_no_op_regression():
    field = _count_field_with_exposure()
    field.exposure[:] = np.nan  # remove all exposure
    field = LDOField(variables=field.variables, space_ids=field.space_ids, time_ids=field.time_ids,
                     X=field.X, W=field.W, resolution="year", exposure=None)
    run = run_ldo(field, K=2, n_subsamples=4, run_residual_scan=False, seed=0)
    assert run.diagnostics["count_exposure_variables"] == 0
