"""Artifact-driven mass-preserving historical municipality geneallocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pegasus.geo.geodata import (
    GeoArtifact,
    GeoArtifactError,
    GeoTransformManifest,
    load_geo_artifact,
    load_geo_transform_manifest,
)


@dataclass(frozen=True)
class GeneallocationResult:
    frame: pl.DataFrame
    artifact: GeoArtifact
    input_total: float
    output_total: float
    conserved: bool
    manifest: GeoTransformManifest


def geneallocate_measure(
    frame: pl.DataFrame,
    *,
    weights_path: str | Path,
    value_column: str,
    source_column: str = "source_id",
    year_column: str = "year",
    aggregation: str = "additive",
    tolerance: float = 1e-9,
    manifest_path: str | Path | None = None,
    require_official: bool = False,
) -> GeneallocationResult:
    if aggregation != "additive":
        raise GeoArtifactError("direct geneallocation of rates/intensive fields is forbidden")
    weights, artifact = load_geo_artifact(
        weights_path,
        required_columns={source_column, year_column, "target_id", "weight"},
    )
    if weights.filter((pl.col("weight") < 0) | (~pl.col("weight").is_finite())).height:
        raise GeoArtifactError("geneallocation weights must be finite and nonnegative")
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
    manifest = load_geo_transform_manifest(
        artifact,
        manifest_path=manifest_path,
        source_geography=source_column,
        target_geography="target_id",
        tolerance=tolerance,
        require_official=require_official,
    )
    return GeneallocationResult(output, artifact, input_total, output_total, conserved, manifest)


def geneallocate_rate_from_components(
    numerator: pl.DataFrame,
    denominator: pl.DataFrame,
    *,
    weights_path: str | Path,
    numerator_column: str = "numerator",
    denominator_column: str = "denominator",
    rate_column: str = "rate",
    source_column: str = "source_id",
    year_column: str = "year",
    manifest_path: str | Path | None = None,
    require_official: bool = False,
) -> pl.DataFrame:
    num = geneallocate_measure(
        numerator,
        weights_path=weights_path,
        value_column=numerator_column,
        source_column=source_column,
        year_column=year_column,
        manifest_path=manifest_path,
        require_official=require_official,
    ).frame
    den = geneallocate_measure(
        denominator,
        weights_path=weights_path,
        value_column=denominator_column,
        source_column=source_column,
        year_column=year_column,
        manifest_path=manifest_path,
        require_official=require_official,
    ).frame
    joined = num.join(den, on=[year_column, "target_id"], how="inner")
    if joined.height != num.height or joined.height != den.height:
        raise GeoArtifactError("transformed numerator and denominator supports do not align")
    return joined.with_columns(
        pl.when(pl.col(denominator_column) > 0)
        .then(pl.col(numerator_column) / pl.col(denominator_column))
        .otherwise(None)
        .alias(rate_column)
    )
