"""MeasuredQuantity — the EFG terminal object (MSD-III §II.3, EFG-OUT-01).

The EFG's terminal output is no longer a materialized *rate*. The denominator
principle (I.2) says normalization belongs in the model as an *offset*:

    log E[count] = log(exposure) + effects        ⇒ rate = exp(effects) is the *implied* view.

So a normalized quantity is emitted as a **count with an exposure**, not a pre-divided
ratio — materializing the rate throws away the count-variance and forces the LDO to
reverse-engineer it. This makes the EFG and the LDO speak one language: count + exposure
+ structure. The rate remains available as the derived ``implied_rate`` view (for
dashboards), never as the modeling input.

This is an additive sidecar: the RN operator still writes its rate parquet unchanged;
alongside it we emit the MeasuredQuantity table + a support reference, so downstream
consumers (the LDO count-with-offset margin, §III.5) can model count + exposure directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

NUMERATOR_COLUMN = "numerator_count"
EXPOSURE_COLUMN = "exposure"


@dataclass(frozen=True)
class MeasuredQuantity:
    """A count + its exposure over a structure, with offset semantics and uncertainty."""

    table: pl.DataFrame                 # [<structure keys/strata>, numerator_count, exposure]
    offset_semantics: str               # "log_exposure": the model uses log(exposure) as an offset
    structure: dict[str, Any]           # {keys, strata, axes}
    provenance: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)

    @property
    def numerator_count(self) -> pl.Series:
        return self.table[NUMERATOR_COLUMN]

    @property
    def exposure(self) -> pl.Series:
        return self.table[EXPOSURE_COLUMN]

    def implied_rate(self) -> pl.Series:
        """The DERIVED rate view ``count / exposure`` (null where exposure ≤ 0) — never the
        modeling input, only a display view."""
        return (
            self.table.select(
                pl.when(pl.col(EXPOSURE_COLUMN) > 0)
                .then(pl.col(NUMERATOR_COLUMN) / pl.col(EXPOSURE_COLUMN))
                .otherwise(None)
                .alias("rate")
            )["rate"]
        )


def measured_quantity_from_rn_join(
    joined: pl.DataFrame,
    *,
    keys: list[str],
    strata: list[str],
    field_id: str,
    provenance: list[str] | tuple[str, ...] | None = None,
    denom_fragility: float = 0.0,
    axes: dict[str, Any] | None = None,
    numerator_col: str = "value_numerator",
    exposure_col: str = "value_denominator",
) -> MeasuredQuantity:
    """Build a MeasuredQuantity from the RN operator's numerator⋈denominator join.

    Carries the per-cell count and exposure (NOT their ratio), so the count-variance and
    the offset are preserved for the model (§III.5).
    """
    structure_cols = [c for c in [*keys, *strata] if c in joined.columns]
    table = joined.select([
        *[pl.col(c) for c in structure_cols],
        pl.col(numerator_col).alias(NUMERATOR_COLUMN),
        pl.col(exposure_col).alias(EXPOSURE_COLUMN),
    ])
    return MeasuredQuantity(
        table=table,
        offset_semantics="log_exposure",
        structure={"keys": list(keys), "strata": list(strata), "axes": dict(axes or {})},
        provenance={"field_id": field_id, "operator": "RN", "provenance": list(provenance or [])},
        uncertainty={"denom_fragility": float(denom_fragility)},
    )


def write_measured_quantity(mq: MeasuredQuantity, path: str | Path) -> Path:
    """Write the count+exposure table as a sidecar parquet; return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mq.table.write_parquet(out)
    return out


__all__ = [
    "MeasuredQuantity",
    "measured_quantity_from_rn_join",
    "write_measured_quantity",
    "NUMERATOR_COLUMN",
    "EXPOSURE_COLUMN",
]
