"""External geospatial artifact loading and validation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class GeoTransformManifest:
    crosswalk_id: str
    source_year: int | None
    target_year: int | None
    source_geography: str
    target_geography: str
    allocation_matrix_hash: str
    stochastic_axis: str
    mass_tolerance: float
    provenance: str
    coverage: float
    production_candidate: bool
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        payload = dict(vars(self))
        payload["warnings"] = list(self.warnings)
        return payload


def load_geo_transform_manifest(
    artifact: GeoArtifact,
    *,
    manifest_path: str | Path | None,
    source_geography: str,
    target_geography: str,
    tolerance: float,
    require_official: bool = False,
) -> GeoTransformManifest:
    if manifest_path is None:
        if require_official:
            raise GeoArtifactError("official geospatial transform requires an artifact manifest")
        return GeoTransformManifest(
            crosswalk_id=Path(artifact.path).stem,
            source_year=None,
            target_year=None,
            source_geography=source_geography,
            target_geography=target_geography,
            allocation_matrix_hash=artifact.sha256,
            stochastic_axis="row",
            mass_tolerance=tolerance,
            provenance="provisional_unmanifested",
            coverage=1.0,
            production_candidate=False,
            warnings=("geo_transform_manifest_missing",),
        )
    path = Path(manifest_path)
    if not path.exists():
        raise GeoArtifactError(f"geospatial transform manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "crosswalk_id", "source_geography", "target_geography", "allocation_matrix_hash",
        "stochastic_axis", "mass_tolerance", "provenance", "coverage",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise GeoArtifactError(f"geospatial transform manifest missing keys: {missing}")
    if payload["allocation_matrix_hash"] != artifact.sha256:
        raise GeoArtifactError("geospatial allocation matrix hash does not match manifest")
    if payload["source_geography"] != source_geography or payload["target_geography"] != target_geography:
        raise GeoArtifactError("geospatial manifest support does not match requested transform")
    provenance = str(payload["provenance"])
    production = provenance == "official" and float(payload["coverage"]) >= 1.0
    if require_official and not production:
        raise GeoArtifactError("geospatial transform is not official with complete coverage")
    return GeoTransformManifest(
        crosswalk_id=str(payload["crosswalk_id"]),
        source_year=int(payload["source_year"]) if payload.get("source_year") is not None else None,
        target_year=int(payload["target_year"]) if payload.get("target_year") is not None else None,
        source_geography=source_geography,
        target_geography=target_geography,
        allocation_matrix_hash=artifact.sha256,
        stochastic_axis=str(payload["stochastic_axis"]),
        mass_tolerance=float(payload["mass_tolerance"]),
        provenance=provenance,
        coverage=float(payload["coverage"]),
        production_candidate=production,
        warnings=tuple(str(value) for value in payload.get("warnings", [])),
    )


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
