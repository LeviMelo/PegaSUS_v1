"""Typed coverage manifest (MSD-III §VIII bounded exhaustiveness, §VIII.2(3)).

Bounded exhaustiveness is honest only if what was NOT searched is recorded as a typed
``unsearched`` region rather than hidden. This manifest records the resolutions,
conditioning depth (lag order), and functional forms an LDO run actually exercised, plus
the regions it explicitly did not search and why — so a "no edge found" is never confused
with "not looked for". The sparsity-of-truth assumption is stated, not assumed silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnsearchedRegion:
    kind: str            # 'resolution' | 'conditioning' | 'functional_form' | 'interaction'
    description: str
    reason: str


@dataclass
class CoverageManifest:
    """What an LDO run searched and, explicitly, what it did not."""

    resolution_searched: str
    lag_orders_searched: tuple[int, ...]
    functional_forms_searched: tuple[str, ...]
    unsearched: list[UnsearchedRegion] = field(default_factory=list)
    # §VIII.2: the sparsity-of-truth assumption MUST be STATED (not a hidden bool). A "no edge in
    # an unsearched region" claim rests on this assumption being true, so it is spelled out.
    sparsity_of_truth_assumption: str = (
        "most fine-grained links are null; unsearched regions are assumed edge-free only under "
        "this sparsity-of-truth prior, whose bounded recall is measured by the §VIII.2(2) audit"
    )

    def mark_unsearched(self, kind: str, description: str, reason: str) -> None:
        self.unsearched.append(UnsearchedRegion(kind=kind, description=description, reason=reason))

    def as_manifest(self) -> dict[str, Any]:
        return {
            "resolution_searched": self.resolution_searched,
            "lag_orders_searched": list(self.lag_orders_searched),
            "functional_forms_searched": list(self.functional_forms_searched),
            "unsearched": [
                {"kind": u.kind, "description": u.description, "reason": u.reason}
                for u in self.unsearched
            ],
            "sparsity_of_truth_assumption": self.sparsity_of_truth_assumption,
            "n_unsearched_regions": len(self.unsearched),
        }


def build_coverage_manifest(
    *, resolution: str, K: int, requested_K: int,
    ran_residual_scan: bool, residual_error: str | None,
) -> CoverageManifest:
    """Assemble the manifest from the run's actual coverage.

    Linear conditional dependence up to lag ``K`` is always searched; the nonlinear residual
    (HSIC) functional form is searched iff the scan ran without error. Higher lags, finer
    resolutions, non-additive interactions, and (when the scan could not run) the nonlinear
    form itself are recorded as typed unsearched regions.
    """
    forms = ["linear_conditional_precision"]
    if ran_residual_scan and not residual_error:
        forms.append("nonlinear_residual_hsic")
    manifest = CoverageManifest(
        resolution_searched=resolution,
        lag_orders_searched=tuple(range(0, K + 1)),
        functional_forms_searched=tuple(forms),
    )
    if requested_K > K:
        manifest.mark_unsearched(
            "conditioning", f"lag orders {K + 1}..{requested_K}",
            "adaptive-K reduced the lag depth to fit the panel/envelope",
        )
    manifest.mark_unsearched(
        "resolution", f"resolutions finer than '{resolution}'",
        "the joint precision is fit at one grain; finer detail is shrunk to its parent (§III.3)",
    )
    manifest.mark_unsearched(
        "interaction", "non-additive effect-modification (varying-coefficient) terms",
        "interactions are named terms, not exhaustively enumerated at this pass",
    )
    if not (ran_residual_scan and not residual_error):
        manifest.mark_unsearched(
            "functional_form", "nonlinear residual (HSIC) dependence",
            residual_error or "residual scan not requested",
        )
    return manifest


__all__ = ["CoverageManifest", "UnsearchedRegion", "build_coverage_manifest"]
