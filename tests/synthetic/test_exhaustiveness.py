"""EXH-01 — sensitivity screens + coverage manifest (MSD-III §VIII)."""

from __future__ import annotations

import numpy as np

from pegasus.ldo.exhaustiveness import (
    CoverageManifest,
    should_drill_down,
)


def test_heterogeneity_screen_fires_on_cancellation() -> None:
    # opposite regional effects sum to ~0 → a mean-based test would MISS this link
    effects = np.array([0.6, -0.6, 0.55, -0.55])
    assert abs(effects.mean()) < 0.05                  # the pooled mean hides it
    drill, reason = should_drill_down(effects)
    assert drill and "heterogeneity" in reason         # the sensitivity filter catches it


def test_max_subgroup_screen_fires_on_localized_effect() -> None:
    # one strong subgroup diluted to a weak pooled mean (sparsity dilution)
    effects = np.array([0.0, 0.0, 0.0, 0.9])
    assert abs(effects.mean()) < 0.3
    drill, reason = should_drill_down(effects)
    assert drill and "max_subgroup" in reason


def test_true_null_does_not_drill() -> None:
    effects = np.array([0.02, -0.01, 0.0, 0.01])
    drill, reason = should_drill_down(effects)
    assert not drill and reason == "below_sensitivity_thresholds"


def test_coverage_manifest_types_unsearched_regions() -> None:
    m = CoverageManifest()
    m.mark_searched(region="Alagoas", resolution="chapter")
    m.mark_unsearched(region="Alagoas", resolution="leaf", reason="pruned_below_sensitivity_filter")
    assert m.unsearched[0]["reason"] == "pruned_below_sensitivity_filter"   # typed, never silent
    assert m.sparsity_of_truth_assumption                                   # the assumption is stated
