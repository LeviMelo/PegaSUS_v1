from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NullRegime:
    support_kind: str
    null_strategy: str
    permutation_unit: str
    fdr_method: str
    residual_mode_required: str
    warnings: tuple[str, ...] = ()


def null_regime_for_support(support_kind: str) -> NullRegime:
    if support_kind == "annual_municipal_panel":
        return NullRegime(support_kind, "spatial_block_cyclic_time_shift", "cross_fitted_residual", "BY", "cross_fitted")
    if support_kind == "monthly_seasonal_panel":
        return NullRegime(support_kind, "season_preserving_moving_block_circular_shift", "cross_fitted_residual", "BY", "cross_fitted")
    if support_kind == "facility_stock":
        return NullRegime(support_kind, "restricted_intra_uf_swap", "raw_field", "storey_q", "not_required")
    return NullRegime(support_kind, "geo_adjacency_shuffle", "model_dependent", "BH", "model_dependent")


def assert_monthly_null_preserves_season(null_strategy: str) -> None:
    if null_strategy != "season_preserving_moving_block_circular_shift":
        raise ValueError("monthly_seasonal_panel_null_must_preserve_season")
