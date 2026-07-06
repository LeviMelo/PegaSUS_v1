from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SIDRARegime = Literal[
    "direct",
    "harmonize",
    "deflate",
    "bounded_interpolate",
    "cross_sectional",
    "do_not_reconstruct",
]


@dataclass(frozen=True)
class SIDRAContextRegimeResult:
    regime: SIDRARegime
    stdfm_gate: bool
    warnings: tuple[str, ...]
    reason: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "stdfm_gate": self.stdfm_gate,
            "warnings": list(self.warnings),
            "reason": self.reason,
        }


def classify_sidra_context_regime(
    *,
    missing_t: bool,
    schema_stable: bool,
    schema_mismatch: bool,
    projectable: bool,
    unit: str,
    anchors_bounded: bool,
    concept_compatible: bool,
    temporal_points: int,
    dynamics: str,
) -> SIDRAContextRegimeResult:
    unit_norm = unit.strip().lower()
    if not missing_t and schema_stable:
        return SIDRAContextRegimeResult("direct", False, tuple(), "Complete stable SIDRA segment.")
    if schema_mismatch and projectable:
        return SIDRAContextRegimeResult("harmonize", False, ("sidra_schema_harmonization",), "Projectable schema mismatch.")
    if unit_norm in {"r$", "milr$", "sm"}:
        return SIDRAContextRegimeResult("deflate", False, ("sidra_deflation_required",), "Monetary or salary-minimum SIDRA unit.")
    gate = anchors_bounded and concept_compatible and temporal_points >= 3 and dynamics in {
        "continuous",
        "semi-continuous",
        "proportion",
        "positive",
    }
    if gate:
        return SIDRAContextRegimeResult(
            "bounded_interpolate",
            True,
            ("stdfm_candidate_requires_certification",),
            "SIDRA field meets ST-DFM gate but solver is certification-bound.",
        )
    if temporal_points >= 1:
        return SIDRAContextRegimeResult(
            "cross_sectional",
            False,
            ("sidra_cross_sectional_only",),
            "SIDRA field lacks enough certified longitudinal structure for ST-DFM.",
        )
    return SIDRAContextRegimeResult("do_not_reconstruct", False, ("sidra_do_not_reconstruct",), "SIDRA field lacks reconstructable support.")
