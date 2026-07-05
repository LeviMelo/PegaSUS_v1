"""Physical SIDRA marginalization and classification projection (MSD §2.12.2,
§2.12.3, §3.9.1, §3.9.2).

The pre-existing ``category_maps.bounded_pushforward_scaffold`` and
``projection.project_classification_to_axis`` only *decide* whether a pushforward
or projection is legal and emit a status string; they never touch the fact data.
This module performs the actual tensor operations on a Polars long-form fact
frame:

* additive pushforward         : ``(π_* Y)(l_keep) = Σ_{l_drop} Y(l_keep, l_drop)``
* rate/proportion pushforward  : ``RN(π_* ν, π_* μ)`` — sum numerator and
                                 denominator separately, then divide (direct
                                 marginalization of a ratio is illegal, §3.9.1)
* classification projection    : ``Y_axis = Σ_χ Π_{α,χ} Y_raw`` for additive

PANEL-01 pre-build, unwired — do not reap (PEGASUS_REFACTOR_MASTER_PLAN.md §1c).
                                 measures; rates require numerator/denominator
                                 recovery, §2.12.2.
"""

from __future__ import annotations

import polars as pl


class PushforwardError(ValueError):
    """Raised when a requested marginalization/projection is not legal on data."""


def execute_additive_pushforward(
    frame: pl.DataFrame,
    *,
    value_col: str,
    keep_axes: list[str],
    drop_axes: list[str],
) -> pl.DataFrame:
    """Marginalize an additive measure by summing over ``drop_axes``.

    ``(π_* Y)(l_keep) = Σ_{l_drop} Y(l_keep, l_drop)``.
    """
    missing = [c for c in [value_col, *keep_axes, *drop_axes] if c not in frame.columns]
    if missing:
        raise PushforwardError(f"frame_missing_columns:{missing}")
    if not keep_axes:
        return frame.select(pl.col(value_col).sum().alias(value_col))
    return (
        frame.group_by(keep_axes, maintain_order=True)
        .agg(pl.col(value_col).sum().alias(value_col))
    )


def execute_rate_pushforward(
    frame: pl.DataFrame,
    *,
    numerator_col: str,
    denominator_col: str,
    keep_axes: list[str],
    output_col: str = "rate",
) -> pl.DataFrame:
    """Marginalize a rate/proportion legally via Radon-Nikodym recovery.

    ``RN(π_* ν, π_* μ) = (Σ ν) / (Σ μ)`` — never the mean of per-cell ratios.
    """
    missing = [c for c in [numerator_col, denominator_col, *keep_axes] if c not in frame.columns]
    if missing:
        raise PushforwardError(f"frame_missing_columns:{missing}")
    grouped = (
        frame.group_by(keep_axes, maintain_order=True).agg(
            [
                pl.col(numerator_col).sum().alias("_num"),
                pl.col(denominator_col).sum().alias("_den"),
            ]
        )
        if keep_axes
        else frame.select(
            [
                pl.col(numerator_col).sum().alias("_num"),
                pl.col(denominator_col).sum().alias("_den"),
            ]
        )
    )
    return grouped.with_columns(
        pl.when(pl.col("_den") > 0)
        .then(pl.col("_num") / pl.col("_den"))
        .otherwise(None)
        .alias(output_col)
    )


def apply_classification_projection(
    frame: pl.DataFrame,
    *,
    source_col: str,
    value_col: str,
    weights: dict[str, dict[str, float]],
    target_col: str = "target_axis",
    keep_axes: list[str] | None = None,
    measure_kind: str = "additive",
) -> pl.DataFrame:
    """Project source classification categories onto canonical axis categories.

    ``Y_axis_{α} = Σ_χ Π_{α,χ} Y_raw_{χ}`` for additive measures. Direct
    projection of rates/proportions is illegal (§2.12.2): those must be projected
    through numerator/denominator recovery, which this function refuses to fake.
    """
    if measure_kind in {"rate", "proportion", "percentage"}:
        raise PushforwardError(
            "ratio_projection_requires_numerator_denominator_recovery"
        )
    missing = [c for c in [source_col, value_col, *(keep_axes or [])] if c not in frame.columns]
    if missing:
        raise PushforwardError(f"frame_missing_columns:{missing}")
    weight_rows = [
        {"_src": source, "_tgt": target, "_w": float(weight)}
        for source, targets in weights.items()
        for target, weight in targets.items()
    ]
    if not weight_rows:
        raise PushforwardError("empty_projection_matrix")
    weight_frame = pl.DataFrame(weight_rows)
    group_axes = list(keep_axes or [])
    joined = frame.join(
        weight_frame, left_on=source_col, right_on="_src", how="inner"
    ).with_columns((pl.col(value_col) * pl.col("_w")).alias("_contrib"))
    return (
        joined.group_by([*group_axes, "_tgt"], maintain_order=True)
        .agg(pl.col("_contrib").sum().alias(value_col))
        .rename({"_tgt": target_col})
    )
