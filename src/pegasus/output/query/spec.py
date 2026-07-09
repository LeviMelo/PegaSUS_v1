"""QuerySpec — the typed request for a derived dataset from a run bundle (FEAT-P3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# The quantity kinds the Output Query Layer can serve. rate/standardized_rate require a denominator.
QUERY_KINDS = ("raw_field", "count", "rate", "standardized_rate", "edge")
DENOMINATOR_KINDS = frozenset({"rate", "standardized_rate"})
OUTPUT_FORMATS = ("parquet", "csv")


class QuerySpec(BaseModel):
    """A request for one derived dataset. ``denominator=None`` uses the registry default
    (FEAT-P4); an override MUST be admissible for the quantity. ``standardize`` names a reference
    population (only for ``kind='standardized_rate'``)."""

    model_config = ConfigDict(extra="forbid")

    quantity: str
    kind: str = "rate"
    denominator: str | None = None
    strata: list[str] = Field(default_factory=list)
    standardize: str | None = None
    per: int = 100_000
    fmt: str = "parquet"
    filters: dict = Field(default_factory=dict)

    def requires_denominator(self) -> bool:
        return self.kind in DENOMINATOR_KINDS


__all__ = ["QuerySpec", "QUERY_KINDS", "DENOMINATOR_KINDS", "OUTPUT_FORMATS"]
