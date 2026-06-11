"""Null-regime registry for PIRS HSIC residual scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NullRegime:
    panel_type: str
    null_strategy: str
    permutations: int
    permutation_unit: str
    fdr_method: str
    residual_mode: str
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "panel_type": self.panel_type,
            "null_strategy": self.null_strategy,
            "permutations": self.permutations,
            "permutation_unit": self.permutation_unit,
            "fdr_method": self.fdr_method,
            "residual_mode": self.residual_mode,
            "warnings": list(self.warnings),
        }


NULL_REGIMES: dict[str, NullRegime] = {
    "annual_municipal_panel": NullRegime(
        panel_type="annual_municipal_panel",
        null_strategy="spatial_block_cyclic_time_shift",
        permutations=1000,
        permutation_unit="cross_fitted_residual",
        fdr_method="BY",
        residual_mode="cross_fitted",
    ),
    "monthly_seasonal_panel": NullRegime(
        panel_type="monthly_seasonal_panel",
        null_strategy="season_preserving_moving_block_circular_shift",
        permutations=2000,
        permutation_unit="cross_fitted_residual",
        fdr_method="BY",
        residual_mode="cross_fitted",
        warnings=("monthly_null_preserves_season",),
    ),
    "cross_sectional_census": NullRegime(
        panel_type="cross_sectional_census",
        null_strategy="geo_adjacency_shuffle",
        permutations=1000,
        permutation_unit="raw_or_residual",
        fdr_method="BH",
        residual_mode="model_dependent",
    ),
    "facility_stock": NullRegime(
        panel_type="facility_stock",
        null_strategy="restricted_intra_uf_spatial_swap",
        permutations=5000,
        permutation_unit="raw_field",
        fdr_method="Storey_q",
        residual_mode="not_required",
    ),
    "sparse_stratified": NullRegime(
        panel_type="sparse_stratified",
        null_strategy="bootstrap_within_strata",
        permutations=1000,
        permutation_unit="residual",
        fdr_method="BY",
        residual_mode="cross_fitted_if_feasible",
    ),
}


def select_null_regime(panel_type: str) -> NullRegime:
    try:
        return NULL_REGIMES[panel_type]
    except KeyError as exc:
        raise ValueError(f"unknown HSIC null panel_type: {panel_type}") from exc


def assert_monthly_null_preserves_season(regime: NullRegime) -> None:
    if regime.panel_type == "monthly_seasonal_panel" and "season_preserving" not in regime.null_strategy:
        raise ValueError("monthly HSIC nulls must preserve season")


def descriptive_only_when_insufficient_blocks(*, spatial_blocks: int, temporal_blocks: int) -> bool:
    return spatial_blocks < 5 or temporal_blocks < 5
