"""Artifact-driven mass-preserving historical municipality geneallocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pegasus.geo.geodata import GeoArtifact, GeoArtifactError, load_geo_artifact


@dataclass(frozen=True)
class GeneallocationResult:
    frame: pl.DataFrame
    artifact: GeoArtifact
    input_total: float
    output_total: float
    conserved: bool


def geneallocate_measure(
    frame: pl.DataFrame,
    *,
    weights_path: str | Path,
    value_column: str,
    source_column: str = "source_id",
    year_column: str = "year",
    aggregation: str = "additive",
    tolerance: float = 1e-9,
) -> GeneallocationResult:
    if aggregation != "additive":
        raise GeoArtifactError("direct geneallocation of rates/intensive fields is forbidden")
    weights, artifact = load_geo_artifact(
        weights_path,
        required_columns={source_column, year_column, "target_id", "weight"},
    )
    sums = weights.group_by([source_column, year_column]).agg(pl.col("weight").sum().alias("weight_sum"))
    bad = sums.filter((pl.col("weight_sum") - 1.0).abs() > tolerance)
    if bad.height:
        raise GeoArtifactError("geneallocation weights must sum to one for every source-year")
    joined = frame.join(weights, on=[source_column, year_column], how="left")
    if joined["target_id"].null_count():
        raise GeoArtifactError("geneallocation weights do not cover all input source-year rows")
    output = (
        joined.with_columns((pl.col(value_column) * pl.col("weight")).alias(value_column))
        .group_by([year_column, "target_id"])
        .agg(pl.col(value_column).sum())
        .sort([year_column, "target_id"])
    )
    input_total = float(frame[value_column].sum())
    output_total = float(output[value_column].sum())
    conserved = abs(input_total - output_total) <= tolerance * max(1.0, abs(input_total))
    if not conserved:
        raise GeoArtifactError(f"geneallocation conservation failure: input={input_total} output={output_total}")
    return GeneallocationResult(output, artifact, input_total, output_total, conserved)
