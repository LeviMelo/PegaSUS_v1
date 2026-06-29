from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SpatialEffectMode = Literal["none", "UF_FE", "municipality_FE", "ICAR"]


@dataclass(frozen=True)
class SpatialEffectSelection:
    mode: SpatialEffectMode
    reason: str
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def select_spatial_effect_mode(
    *,
    budget: str,
    time_period_count: int | None,
    spatial_missingness: float | None,
    moran_i: float | None,
    adjacency_available: bool = True,
    moran_near_zero_threshold: float = 0.03,
) -> SpatialEffectSelection:
    """Select the MSD §6.3 spatial-effect mode.

    UF and municipality fixed effects are materialized by design-matrix
    expansion. ICAR is executed downstream by a penalized-IRLS GMRF layer when
    the design manifest declares a municipality adjacency artifact.
    """

    if str(budget) == "fast":
        return SpatialEffectSelection(mode="UF_FE", reason="budget_fast_selects_uf_fixed_effects")

    if moran_i is not None and abs(float(moran_i)) <= float(moran_near_zero_threshold):
        return SpatialEffectSelection(mode="none", reason="moran_i_near_zero")

    t_count = int(time_period_count or 0)
    zeta = float(spatial_missingness) if spatial_missingness is not None else 1.0
    if t_count >= 10 and zeta <= 0.10:
        return SpatialEffectSelection(
            mode="municipality_FE",
            reason="long_panel_low_spatial_missingness_selects_municipality_fixed_effects",
        )

    if not adjacency_available:
        return SpatialEffectSelection(
            mode="municipality_FE",
            reason="short_panel_without_declared_adjacency_selects_executable_municipality_fixed_effects",
            warnings=("icar_requires_declared_municipality_adjacency",),
        )

    return SpatialEffectSelection(
        mode="ICAR",
        reason="short_panel_or_fragile_spatial_support_requires_icar",
        warnings=("icar_requires_declared_municipality_adjacency",),
    )


__all__ = ["SpatialEffectMode", "SpatialEffectSelection", "select_spatial_effect_mode"]
