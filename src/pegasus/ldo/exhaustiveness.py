"""Bounded exhaustiveness — sensitivity screens + coverage manifest (MSD-III §VIII).

Absolute exhaustiveness is impossible (the information ceiling; the combinatorial link
space), so PegaSUS requires *honest bounded-exhaustiveness*. Naïve coarse-screening on the
aggregate mean has real false negatives: aggregation *hides* links via Simpson reversal,
cancellation (opposite regional effects summing to zero), thresholds, and sparsity
dilution. The remedy (§VIII.2):

1. **Screen on sensitivity, not the mean.** Coarse passes screen on statistics designed to
   fire on what aggregation hides — subgroup *heterogeneity*, *dispersion*, *max-subgroup*
   signal — converting the coarse pass from a *test* (with false negatives) into a
   *sensitive filter* tuned for recall.
2. **Typed coverage manifest.** Whatever is not searched is recorded as a typed
   ``unsearched`` region (with the reason it was pruned) — never a silent gap. The
   sparsity-of-truth assumption is stated, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def heterogeneity_screen(subgroup_effects) -> float:
    """Spread (std) of effects across subgroups — fires on *cancellation*: a case whose
    pooled mean is ~0 because subgroup effects have opposite signs still scores high here."""
    e = np.asarray(subgroup_effects, dtype=np.float64)
    return float(e.std()) if e.size else 0.0


def max_subgroup_screen(subgroup_effects) -> float:
    """Largest |subgroup effect| — fires on a *localized* strong effect the pooled mean
    dilutes (sparsity dilution / a rare-code link swamped in its chapter)."""
    e = np.asarray(subgroup_effects, dtype=np.float64)
    return float(np.max(np.abs(e))) if e.size else 0.0


def dispersion_screen(counts) -> float:
    """Index of dispersion (var/mean) — >1 flags overdispersion / clustering / thresholds."""
    c = np.asarray(counts, dtype=np.float64)
    m = c.mean() if c.size else 0.0
    return float(c.var() / m) if m > 0 else 0.0


def should_drill_down(
    subgroup_effects, *, het_threshold: float = 0.15, max_threshold: float = 0.3,
) -> tuple[bool, str]:
    """A *sensitive filter* (tuned for recall), NOT a test of the pooled mean.

    Drills when subgroup heterogeneity OR the max-subgroup signal exceeds threshold — so a
    cancellation/Simpson case (pooled mean ≈ 0) or a localized effect still triggers the
    fine pass. Returns ``(drill, reason)``.
    """
    het = heterogeneity_screen(subgroup_effects)
    mx = max_subgroup_screen(subgroup_effects)
    reasons = []
    if het >= het_threshold:
        reasons.append(f"heterogeneity={het:.3f}")
    if mx >= max_threshold:
        reasons.append(f"max_subgroup={mx:.3f}")
    return (bool(reasons), "; ".join(reasons) if reasons else "below_sensitivity_thresholds")


@dataclass
class CoverageManifest:
    """Typed record of what was searched vs left ``unsearched`` (§VIII.2)."""

    searched: list[dict] = field(default_factory=list)
    unsearched: list[dict] = field(default_factory=list)     # each: {region, resolution, reason}
    sparsity_of_truth_assumption: str = (
        "most fine-grained links are null; the coarse sensitivity filter has bounded recall"
    )
    random_audit_false_negative_rate: float | None = None

    def mark_unsearched(self, *, region: str, resolution: str, reason: str) -> None:
        self.unsearched.append({"region": region, "resolution": resolution, "reason": reason})

    def mark_searched(self, *, region: str, resolution: str) -> None:
        self.searched.append({"region": region, "resolution": resolution})


__all__ = [
    "heterogeneity_screen", "max_subgroup_screen", "dispersion_screen",
    "should_drill_down", "CoverageManifest",
]
