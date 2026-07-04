"""T0-4 — Inference baseline + output contract (MSD-III §III, X.1).

The guardrail behind which Phase 4 strangles the PIRS slice-zoo into the LDO.
Pins the *current* inference behavior so `run_ldo` must reproduce it before the
old scanner is retired:

    * HSIC mode selection is exact across its decision thresholds
    * GLM family routing is exact for count / proportion / simplex / skewed / default
    * spatial-effect-mode precedence is exact
    * the 17-key output bundle exists (Hypotheses is first-class)
    * empty first-class keys are governed by a typed (run_profile, execution_stage) contract
"""

from __future__ import annotations

import pytest

from pegasus.output.schemas import (
    INFERENCE_KEYS,
    LEGACY_INFERENCE_KEYS,
    OUTPUT_BUNDLE_FILES,
    OUTPUT_KEYS,
    required_nonempty_keys,
)
from pegasus.pirs.families import family_for_outcome
from pegasus.pirs.hsic import select_hsic_mode
from pegasus.pirs.schemas import FieldCandidate
from pegasus.pirs.spatial import select_spatial_effect_mode


def _cand(**overrides) -> FieldCandidate:
    params = dict(
        field_id="f",
        role="outcome",
        utility=1.0,
        q_state="warning",
        carrier="gaussian_measure",
        unit="value",
        support={},
    )
    params.update(overrides)
    return FieldCandidate(**params)


# --------------------------------------------------------------------------- #
# 1. HSIC mode selection is exact.
# --------------------------------------------------------------------------- #
def test_hsic_mode_selection_exact() -> None:
    assert select_hsic_mode(n_eff=50, budget="standard") == "disabled"       # n_eff < 100
    assert select_hsic_mode(n_eff=1000, budget="standard") == "exact"        # small enough for exact
    assert select_hsic_mode(n_eff=6000, budget="standard") == "nystrom"      # large + standard/deep
    assert select_hsic_mode(n_eff=6000, budget="deep") == "nystrom"
    assert select_hsic_mode(n_eff=6000, budget="fast") == "rff"              # large + fast
    assert select_hsic_mode(n_eff=1000, budget="standard", user_disabled=True) == "disabled"
    assert (
        select_hsic_mode(n_eff=1000, budget="standard", cuda_required=True, cuda_available=False)
        == "cuda_unavailable_abort"
    )


# --------------------------------------------------------------------------- #
# 2. GLM family routing is exact.
# --------------------------------------------------------------------------- #
def test_family_routing() -> None:
    offset = _cand(field_id="pop", role="offset", carrier="population_denominator")

    # count with exposure → Poisson with log offset
    count = _cand(carrier="event_count", unit="count", variance=None)
    assert family_for_outcome(outcome=count, offset=offset) == "poisson_count_with_log_offset"

    # proportion → binomial; overdispersed → beta-binomial
    prop = _cand(carrier="proportion", unit="proportion")
    assert family_for_outcome(outcome=prop) == "binomial_proportion"
    prop_od = _cand(carrier="proportion", unit="proportion", warnings=("overdispersed",))
    assert family_for_outcome(outcome=prop_od) == "beta_binomial"

    # simplex/composition → Dirichlet
    assert family_for_outcome(outcome=_cand(carrier="simplex")) == "dirichlet"

    # skewed positive → lognormal; with a zero mass → two-part lognormal
    assert family_for_outcome(outcome=_cand(carrier="skewed_positive")) == "lognormal"
    assert (
        family_for_outcome(outcome=_cand(carrier="skewed_positive", warnings=("zero_mass",)))
        == "two_part_lognormal"
    )

    # default fallback
    assert family_for_outcome(outcome=_cand(carrier="gaussian_measure")) == "gaussian_identity"

    # a rate-with-population-denominator outcome is blocked (must model count+exposure)
    with pytest.raises(ValueError):
        family_for_outcome(outcome=_cand(carrier="deaths/population"))


# --------------------------------------------------------------------------- #
# 3. Spatial-effect-mode precedence is exact.
# --------------------------------------------------------------------------- #
def test_spatial_mode_precedence() -> None:
    # budget=fast dominates everything
    fast = select_spatial_effect_mode(budget="fast", time_period_count=1, spatial_missingness=0.5, moran_i=0.9)
    assert fast.mode == "UF_FE"

    # near-zero Moran → no spatial effect
    flat = select_spatial_effect_mode(budget="standard", time_period_count=12, spatial_missingness=0.0, moran_i=0.01)
    assert flat.mode == "none"

    # long panel + low spatial missingness → municipality fixed effects
    longpanel = select_spatial_effect_mode(budget="standard", time_period_count=12, spatial_missingness=0.05, moran_i=0.5)
    assert longpanel.mode == "municipality_FE"

    # short panel, no adjacency declared → executable municipality FE (ICAR needs adjacency)
    no_adj = select_spatial_effect_mode(
        budget="standard", time_period_count=3, spatial_missingness=0.5, moran_i=0.5, adjacency_available=False
    )
    assert no_adj.mode == "municipality_FE"

    # short panel + adjacency + real autocorrelation → ICAR
    icar = select_spatial_effect_mode(
        budget="standard", time_period_count=3, spatial_missingness=0.5, moran_i=0.5, adjacency_available=True
    )
    assert icar.mode == "ICAR"


# --------------------------------------------------------------------------- #
# 4. The 17-key output bundle exists; Hypotheses is first-class.
# --------------------------------------------------------------------------- #
def test_seventeen_key_bundle() -> None:
    assert len(OUTPUT_BUNDLE_FILES) == 17
    assert len(OUTPUT_KEYS) == 17
    assert "Hypotheses" in OUTPUT_BUNDLE_FILES


# --------------------------------------------------------------------------- #
# 5. Empty first-class keys carry a typed (run_profile, execution_stage) reason.
# --------------------------------------------------------------------------- #
def test_empty_keys_typed_by_profile_and_stage() -> None:
    # Hypotheses (the LDO output) is required only at investigate...
    assert "Hypotheses" in required_nonempty_keys("contextual", "investigate")
    assert "Hypotheses" not in required_nonempty_keys("contextual", "compile")

    # ...and legacy pairwise tables are never required (LinkRecord is authoritative).
    for stage in ("compile", "investigate"):
        req = required_nonempty_keys("full", stage)
        assert not (LEGACY_INFERENCE_KEYS & req)

    # inference keys are all stage-gated away from validate/compile
    assert not (INFERENCE_KEYS & required_nonempty_keys("full", "validate"))
