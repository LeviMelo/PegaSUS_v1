from __future__ import annotations

from pegasus.sidra.schemas import BoundedPushforwardResult


def bounded_pushforward_scaffold(
    *,
    raw_axes: list[str],
    demanded_axes: list[str],
    aggregation: str,
    high_dimensional: bool,
) -> BoundedPushforwardResult:
    raw = list(dict.fromkeys(raw_axes))
    demanded = list(dict.fromkeys(demanded_axes))
    drop = [axis for axis in raw if axis not in demanded]

    if not high_dimensional and not drop:
        return BoundedPushforwardResult(
            status="not_required",
            axes_kept=demanded,
            axes_dropped=[],
            warnings=[],
        )

    if aggregation not in {"additive", "compositional"}:
        return BoundedPushforwardResult(
            status="blocked",
            axes_kept=demanded,
            axes_dropped=drop,
            warnings=["bounded_pushforward_illegal_for_aggregation"],
            reason="High-dimensional bounding requires additive/compositional pushforward or denominator recovery.",
        )

    return BoundedPushforwardResult(
        status="bounded",
        axes_kept=demanded,
        axes_dropped=drop,
        warnings=["high_dimensional_bounded_pushforward"],
    )
