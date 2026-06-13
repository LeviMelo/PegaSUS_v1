"""Null-regime registry for PIRS HSIC residual scans."""

from __future__ import annotations

from dataclasses import dataclass
import random
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


def generate_null_indices(
    *,
    strategy: str,
    n: int,
    support: dict[str, Any],
    rng: random.Random,
) -> list[int]:
    if n <= 0:
        return []
    if strategy in {"unrestricted_permutation", "permutation_linear_centered"}:
        indices = list(range(n))
        rng.shuffle(indices)
        return indices
    if strategy == "spatial_block_cyclic_time_shift":
        shape = support.get("panel_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2 or int(shape[0]) * int(shape[1]) != n:
            raise ValueError("annual panel HSIC null requires panel_shape=[space,time].")
        space, time = int(shape[0]), int(shape[1])
        blocks = support.get("spatial_blocks")
        if not isinstance(blocks, (list, tuple)) or len(blocks) != space:
            raise ValueError("annual panel HSIC null requires one spatial_blocks label per space unit.")
        block_values = sorted(set(str(value) for value in blocks))
        shuffled = block_values[:]
        rng.shuffle(shuffled)
        block_map = dict(zip(block_values, shuffled, strict=True))
        members = {block: [i for i, value in enumerate(blocks) if str(value) == block] for block in block_values}
        output = [0] * n
        for target_space in range(space):
            source_block = block_map[str(blocks[target_space])]
            candidates = members[source_block]
            source_space = candidates[target_space % len(candidates)]
            shift = rng.randrange(1, time) if time > 1 else 0
            for t in range(time):
                output[target_space * time + t] = source_space * time + ((t + shift) % time)
        return output
    if strategy == "season_preserving_moving_block_circular_shift":
        shape = support.get("panel_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2 or int(shape[0]) * int(shape[1]) != n:
            raise ValueError("monthly HSIC null requires panel_shape=[space,time].")
        space, time = int(shape[0]), int(shape[1])
        if time < 12 or time % 12 != 0:
            raise ValueError("monthly HSIC null requires a whole number of 12-month seasons.")
        years = time // 12
        output = [0] * n
        for s in range(space):
            year_shift = rng.randrange(1, years) if years > 1 else 0
            for t in range(time):
                month = t % 12
                year = t // 12
                source_t = ((year + year_shift) % years) * 12 + month
                output[s * time + t] = s * time + source_t
        return output
    if strategy in {"geo_adjacency_shuffle", "restricted_intra_uf_spatial_swap", "bootstrap_within_strata"}:
        key = {
            "geo_adjacency_shuffle": "adjacency_components",
            "restricted_intra_uf_spatial_swap": "uf_strata",
            "bootstrap_within_strata": "strata",
        }[strategy]
        labels = support.get(key)
        if not isinstance(labels, (list, tuple)) or len(labels) != n:
            raise ValueError(f"{strategy} requires support.{key} with n labels.")
        output = list(range(n))
        for label in sorted(set(str(value) for value in labels)):
            positions = [i for i, value in enumerate(labels) if str(value) == label]
            if strategy == "bootstrap_within_strata":
                sampled = [rng.choice(positions) for _ in positions]
            else:
                sampled = positions[:]
                rng.shuffle(sampled)
            for target, source in zip(positions, sampled, strict=True):
                output[target] = source
        return output
    raise ValueError(f"unknown HSIC null strategy: {strategy}")
