"""Minimum Comparable Area contraction for additive measures."""

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
class AMCResult:
    frame: pl.DataFrame
    artifact: GeoArtifact
    input_total: float
    output_total: float
    conserved: bool
    manifest: GeoTransformManifest


def contract_to_amc(
    frame: pl.DataFrame,
    *,
    crosswalk_path: str | Path,
    value_column: str,
    municipality_column: str = "municipality_id",
    year_column: str = "year",
    aggregation: str = "additive",
    tolerance: float = 1e-9,
    manifest_path: str | Path | None = None,
    require_official: bool = False,
) -> AMCResult:
    if aggregation != "additive":
        raise GeoArtifactError("AMC contraction of rates/intensive fields is illegal; contract measures and denominators separately")
    crosswalk, artifact = load_geo_artifact(
        crosswalk_path,
        required_columns={municipality_column, year_column, "amc_id"},
    )
    if crosswalk.select([municipality_column, year_column]).is_duplicated().any():
        raise GeoArtifactError("AMC crosswalk is not functional by municipality-year")
    joined = frame.join(crosswalk.select([municipality_column, year_column, "amc_id"]), on=[municipality_column, year_column], how="left")
    if joined["amc_id"].null_count():
        raise GeoArtifactError("AMC crosswalk does not cover all input municipality-year rows")
    group_columns = [column for column in frame.columns if column not in {municipality_column, value_column}]
    group_columns = group_columns + ["amc_id"]
    group_columns = list(dict.fromkeys(group_columns))
    output = joined.group_by(group_columns).agg(pl.col(value_column).sum()).sort(group_columns)
    input_total = float(frame[value_column].sum())
    output_total = float(output[value_column].sum())
    conserved = abs(input_total - output_total) <= tolerance * max(1.0, abs(input_total))
    if not conserved:
        raise GeoArtifactError(f"AMC conservation failure: input={input_total} output={output_total}")
    manifest = load_geo_transform_manifest(
        artifact,
        manifest_path=manifest_path,
        source_geography=municipality_column,
        target_geography="amc_id",
        tolerance=tolerance,
        require_official=require_official,
    )
    return AMCResult(output, artifact, input_total, output_total, conserved, manifest)
