from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegasus.sidra.category_maps import bounded_pushforward_scaffold


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
