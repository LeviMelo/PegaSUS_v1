# PANEL-01 pre-build, unwired — do not reap (PEGASUS_REFACTOR_MASTER_PLAN.md §1c).
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pegasus.sources.sidra.category_maps import bounded_pushforward_scaffold

if TYPE_CHECKING:
    import polars as pl


class HighDimensionalExposureError(ValueError):
    """Raised when high-dimensional SIDRA exposure is requested without bounded pushforward."""


@dataclass(frozen=True)
class HighDimensionalBound:
    status: str
    raw_axes: tuple[str, ...]
    exposed_axes: tuple[str, ...]
    axes_dropped: tuple[str, ...]
    estimated_cells_raw: int
    estimated_cells_bounded: int
    warnings: tuple[str, ...]
    reason: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "raw_axes": list(self.raw_axes),
            "exposed_axes": list(self.exposed_axes),
            "axes_dropped": list(self.axes_dropped),
            "estimated_cells_raw": self.estimated_cells_raw,
            "estimated_cells_bounded": self.estimated_cells_bounded,
            "warnings": list(self.warnings),
            "reason": self.reason,
        }


def _estimate_cells(axis_cardinalities: dict[str, int], axes: list[str]) -> int:
    cells = 1
    for axis in axes:
        cells *= int(axis_cardinalities.get(axis, 1))
    return cells


def bound_high_dimensional_sidra_exposure(
    *,
    raw_axes: list[str],
    demanded_axes: list[str],
    axis_cardinalities: dict[str, int],
    aggregation: str,
    high_dimensional: bool,
) -> HighDimensionalBound:
    result = bounded_pushforward_scaffold(
        raw_axes=raw_axes,
        demanded_axes=demanded_axes,
        aggregation=aggregation,
        high_dimensional=high_dimensional,
    )
    raw_cells = _estimate_cells(axis_cardinalities, raw_axes)
    bounded_cells = _estimate_cells(axis_cardinalities, result.axes_kept)
    return HighDimensionalBound(
        status=result.status,
        raw_axes=tuple(raw_axes),
        exposed_axes=tuple(result.axes_kept),
        axes_dropped=tuple(result.axes_dropped),
        estimated_cells_raw=raw_cells,
        estimated_cells_bounded=bounded_cells,
        warnings=tuple(result.warnings),
        reason=result.reason,
    )


def require_bounded_pushforward(bound: HighDimensionalBound) -> None:
    if bound.status == "blocked":
        raise HighDimensionalExposureError(bound.reason or "High-dimensional SIDRA exposure is not legally bounded.")


def execute_bounded_pushforward(
    frame: "pl.DataFrame",
    bound: HighDimensionalBound,
    *,
    value_col: str,
    numerator_col: str | None = None,
    denominator_col: str | None = None,
) -> "pl.DataFrame":
    """Physically marginalize a SIDRA fact frame onto the bounded axis set.

    This is the real groupby-sum the scaffold only ever decided about. For
    additive measures it sums ``value_col`` over the dropped axes; for
    rates/proportions it requires numerator/denominator columns and recovers the
    ratio via Radon-Nikodym (direct ratio marginalization is illegal, §3.9.1).
    """
    require_bounded_pushforward(bound)
    if bound.status == "not_required":
        return frame
    from pegasus.sources.sidra.pushforward import (
        execute_additive_pushforward,
        execute_rate_pushforward,
    )

    keep = list(bound.exposed_axes)
    drop = list(bound.axes_dropped)
    if numerator_col is not None and denominator_col is not None:
        return execute_rate_pushforward(
            frame,
            numerator_col=numerator_col,
            denominator_col=denominator_col,
            keep_axes=keep,
        )
    return execute_additive_pushforward(
        frame, value_col=value_col, keep_axes=keep, drop_axes=drop
    )
