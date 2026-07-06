"""T0-3 — RaceBridge numerics contract (MSD-III §II.5, X.1; pre-redesign pin).

Pins the *current* Bayesian ecological RaceBridge numerics so the Phase-2 redesign
(RACE-01..07: per-source, literature-informed emission matrices) cannot silently
change the load-bearing math. Four contracts (TDD §5, T0-3):

    1. the reallocation weight W depends on the local self-declared composition π
    2. race_bridge_cv is dispersion ACROSS posterior draws, not cross-category spread
    3. sensitivity_width is derived from the inf/sup of the credible set
    4. bridge posteriors are typed uncertain (bridge-derived → state ≤ fragile)
"""

from __future__ import annotations

import pytest

from pegasus.measurement.race import (
    RaceBridgeCounts,
    fixedc_dynamic_weight_bridge,
    load_race_bridge_prior,
)

_PRIOR = "config/priors/race_bridge/fixedC_sim_admin_to_ibge_selfdeclared_v1.json"


def _counts(shares: dict | None = None) -> RaceBridgeCounts:
    support: dict = {"n_events": 10}
    if shares is not None:
        support["target_population_shares"] = shares
    return RaceBridgeCounts(
        raw_admin_counts={"1": 5, "2": 5, "3": 0, "4": 0, "5": 0},
        missing_count=0,
        total_count=10,
        support=support,
    )


# --------------------------------------------------------------------------- #
# 1. W depends on local π: perturbing the local composition changes the posterior.
# --------------------------------------------------------------------------- #
def test_weight_depends_on_local_pi() -> None:
    prior = load_race_bridge_prior(_PRIOR)
    parda_heavy = {"branca": 0.1, "preta": 0.1, "amarela": 0.1, "parda": 0.6, "indigena": 0.1}
    branca_heavy = {"branca": 0.6, "preta": 0.1, "amarela": 0.1, "parda": 0.1, "indigena": 0.1}

    post_parda = fixedc_dynamic_weight_bridge(_counts(parda_heavy), prior)
    post_branca = fixedc_dynamic_weight_bridge(_counts(branca_heavy), prior)

    # Same administrative input, different local π → different self-declared posterior.
    assert post_parda.posterior_counts["parda"] > post_branca.posterior_counts["parda"]
    assert post_parda.effective_bridge_mode == "localPi_posteriorC"


# --------------------------------------------------------------------------- #
# 2. race_bridge_cv is across-draw dispersion, not cross-category spread.
# --------------------------------------------------------------------------- #
def test_cv_is_draw_dispersion_not_category_spread() -> None:
    prior = load_race_bridge_prior(_PRIOR)
    posterior = fixedc_dynamic_weight_bridge(_counts(), prior)

    values = list(posterior.posterior_counts.values())
    mean = sum(values) / len(values)
    cross_category_cv = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 / mean

    assert posterior.race_bridge_cv > 0
    assert posterior.race_bridge_cv != pytest.approx(cross_category_cv), (
        "race_bridge_cv must be posterior-draw dispersion (uncertainty), "
        "never the spread across race categories."
    )


# --------------------------------------------------------------------------- #
# 3. sensitivity_width comes from the inf/sup of the credible set.
# --------------------------------------------------------------------------- #
def test_sensitivity_width_from_credible_set() -> None:
    prior = load_race_bridge_prior(_PRIOR)
    posterior = fixedc_dynamic_weight_bridge(_counts({"branca": 0.2, "preta": 0.2, "amarela": 0.2, "parda": 0.2, "indigena": 0.2}), prior)

    assert 0.0 < posterior.sensitivity_width <= 1.0
    # the credible set brackets the point estimate for every target category
    for target, point in posterior.posterior_counts.items():
        assert posterior.lower_counts[target] <= point + 1e-9
        assert posterior.upper_counts[target] >= point - 1e-9
        # a nonzero interval exists somewhere → sensitivity_width is nontrivial
    interval_seen = any(
        posterior.upper_counts[t] - posterior.lower_counts[t] > 1e-9
        for t in posterior.posterior_counts
    )
    assert interval_seen


# --------------------------------------------------------------------------- #
# 4. Bridge posteriors are typed uncertain (bridge-derived → state ≤ fragile).
# --------------------------------------------------------------------------- #
def test_posterior_typed_uncertain_and_bridge_derived() -> None:
    prior = load_race_bridge_prior(_PRIOR)
    posterior = fixedc_dynamic_weight_bridge(_counts(), prior)
    meta = posterior.metadata()

    # the not-raw-observation warning is the marker the count executor uses to set
    # state="fragile" / dashboard_safe="warning" (efg/dag.py) — never verified.
    assert "bayesian_ecological_bridge_warning" in meta
    assert "bridge-derived" in meta["bayesian_ecological_bridge_warning"]
    assert meta["race_bridge_cv"] == posterior.race_bridge_cv
    assert meta["sensitivity_width"] == posterior.sensitivity_width
    # administrative race is preserved, never overwritten (the §4 epistemic guarantee)
    assert posterior.raw_admin_counts == {"1": 5, "2": 5, "3": 0, "4": 0, "5": 0}
