"""StARS λ₁ selection (Theme-4) — the selected penalty is the largest stable graph.

Plant a couple of contemporaneous edges among many independent noise variables; the
StARS-selected λ₁ must recover the planted edges at low instability, a too-small λ₁
must be unstable (ξ > beta), and select_lambda=False must leave run_ldo untouched.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import LDOField
from pegasus.ldo.lambda_select import select_lambda_stars
from pegasus.ldo.margins import GaussianField, gaussianize_field
from pegasus.ldo.orchestrator import run_ldo


def _planted_field(*, S: int = 60, T: int = 12, n_noise: int = 8, seed: int = 0) -> GaussianField:
    """Two strong contemporaneous edges A–B and C–D; the rest independent noise."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((S, T))
    B = 0.9 * A + 0.3 * rng.standard_normal((S, T))
    C = rng.standard_normal((S, T))
    D = 0.9 * C + 0.3 * rng.standard_normal((S, T))
    noise = [rng.standard_normal((S, T)) for _ in range(n_noise)]
    X = np.stack([A, B, C, D, *noise], axis=0)
    variables = ("A", "B", "C", "D") + tuple(f"noise{i}" for i in range(n_noise))
    return LDOField(
        variables=variables,
        space_ids=tuple(str(270000 + i) for i in range(S)),
        time_ids=tuple(range(T)),
        X=X,
        W=np.ones_like(X),
        resolution="year",
    )


def _selected_pairs(field: GaussianField, lam: float) -> set:
    from pegasus.ldo.lags import fit_lagged_links
    fit = fit_lagged_links(field, K=1, lambda1=lam)
    return {tuple(sorted((s, t))) for s, t, _ in fit.contemporaneous}


def test_stars_selects_largest_stable_graph() -> None:
    gf = gaussianize_field(_planted_field(), seed=0)
    grid = np.geomspace(0.02, 0.5, 7)

    lam_star, report = select_lambda_stars(
        gf, grid, K=1, n_subsamples=10, subsample_frac=0.7, seed=1, beta=0.05,
    )

    # the selected λ recovers both planted edges...
    pairs = _selected_pairs(gf, lam_star)
    assert ("A", "B") in pairs and ("C", "D") in pairs, f"planted edges missing at λ*={lam_star}: {pairs}"

    # ...at instability under the bound...
    xi = {l: x for l, x in zip(report.lambda_grid, report.instability)}
    assert xi[lam_star] <= 0.05, f"selected λ* instability {xi[lam_star]} exceeds beta"

    # ...while the smallest (densest) λ on the grid is unstable (ξ > beta).
    lam_min = min(report.lambda_grid)
    assert xi[lam_min] > 0.05, f"densest λ={lam_min} should be unstable; ξ={xi[lam_min]}"
    assert lam_star > lam_min  # StARS did not pick the raw densest


def test_select_lambda_default_off_is_noop() -> None:
    field = _planted_field(seed=3)
    base = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0)
    assert base.diagnostics["lambda_selection"] is None

    # explicit select_lambda=False is byte-identical to the default
    off = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0, select_lambda=False)
    assert off.diagnostics["lambda_selection"] is None
    assert [r.as_row() for r in off.link_records] == [r.as_row() for r in base.link_records]

    # turning it on populates the diagnostic without disturbing the default path
    on = run_ldo(field, K=1, n_subsamples=6, run_residual_scan=False, seed=0, select_lambda=True)
    assert on.diagnostics["lambda_selection"] is not None
    assert on.diagnostics["lambda_selection"]["method"] == "stars"
