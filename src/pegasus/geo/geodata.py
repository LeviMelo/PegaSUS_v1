"""External geospatial artifact loading and validation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pegasus.core.hashing import sha256_file


class GeoArtifactError(ValueError):
    """Raised when required official/external geospatial evidence is unavailable."""


@dataclass(frozen=True)
class GeoArtifact:
    path: str
    sha256: str
    row_count: int
    columns: tuple[str, ...]


def load_geo_artifact(path: str | Path, *, required_columns: set[str]) -> tuple[pl.DataFrame, GeoArtifact]:
    source = Path(path)
    if not source.exists():
        raise GeoArtifactError(f"required geospatial artifact is missing: {source}")
    if source.suffix.lower() == ".parquet":
        frame = pl.read_parquet(source)
    elif source.suffix.lower() in {".csv", ".tsv"}:
        frame = pl.read_csv(source, separator="\t" if source.suffix.lower() == ".tsv" else ",")
    else:
        raise GeoArtifactError(f"unsupported geospatial artifact format: {source.suffix}")
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise GeoArtifactError(f"geospatial artifact missing columns {missing}: {source}")
    return frame, GeoArtifact(str(source), sha256_file(source), frame.height, tuple(frame.columns))
