"""Source artifact reality contracts for DATASUS/SIDRA compiler inputs.

Slice 12A intentionally does not perform live acquisition. It defines the hard
boundary that lets the compiler and audits distinguish fixtures from materialized
source artifacts before compile integration is widened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json

import polars as pl

SOURCE_ARTIFACT_SCHEMA_VERSION = "1.0"

ALLOWED_SOURCE_SYSTEMS = {
    "SIM-DO",
    "SINASC",
    "CNES-ST",
    "SIH-RD",
    "SIDRA",
    "IBGE-SIDRA",
}

ALLOWED_ARTIFACT_ROLES = {
    "raw_payload",
    "raw_table",
    "processed_events",
    "normalized_facts",
    "request_manifest",
    "metadata_table",
    "extraction_log",
}

ALLOWED_PROVENANCE_MODES = {
    "fixture",
    "cached_external",
    "materialized_external",
}


class SourceArtifactError(ValueError):
    """Raised when source artifacts cannot satisfy data-layer reality contracts."""


@dataclass(frozen=True)
class SourceArtifact:
    source_system: str
    artifact_role: str
    path: str
    content_hash: str
    byte_size: int
    row_count: int | None
    columns: list[str]
    provenance_mode: str
    source_manifest_hash: str | None
    manifest_path: str | None
    warnings: list[str]
    inspected_at: str

    def as_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_shape(path: Path) -> tuple[int | None, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(path)
        return df.height, list(df.columns)
    if suffix in {".csv", ".tsv"}:
        separator = "	" if suffix == ".tsv" else ","
        df = pl.read_csv(path, separator=separator)
        return df.height, list(df.columns)
    if suffix in {".json", ".jsonl"}:
        return None, []
    return None, []


def _as_required_set(values: Iterable[str] | None) -> set[str]:
    return {str(v) for v in values or [] if str(v)}


def inspect_source_artifact(
    *,
    path: str | Path,
    source_system: str,
    artifact_role: str,
    provenance_mode: str,
    source_manifest_hash: str | None = None,
    manifest_path: str | Path | None = None,
    required_columns: Iterable[str] | None = None,
) -> SourceArtifact:
    artifact_path = Path(path)
    if source_system not in ALLOWED_SOURCE_SYSTEMS:
        raise SourceArtifactError(f"Unsupported source system: {source_system!r}")
    if artifact_role not in ALLOWED_ARTIFACT_ROLES:
        raise SourceArtifactError(f"Unsupported artifact role: {artifact_role!r}")
    if provenance_mode not in ALLOWED_PROVENANCE_MODES:
        raise SourceArtifactError(f"Unsupported provenance mode: {provenance_mode!r}")
    if not artifact_path.exists():
        raise SourceArtifactError(f"Source artifact path does not exist: {artifact_path}")
    if artifact_path.is_dir():
        raise SourceArtifactError(f"Source artifact path is a directory, not a file: {artifact_path}")

    row_count, columns = _table_shape(artifact_path)
    required = _as_required_set(required_columns)
    missing_columns = sorted(required.difference(columns))
    if missing_columns:
        raise SourceArtifactError(
            f"Source artifact {artifact_path} is missing required columns: {missing_columns}"
        )

    warnings: list[str] = []
    if provenance_mode == "fixture":
        warnings.append("fixture_source_artifact_not_production_candidate")
    if provenance_mode != "fixture" and not source_manifest_hash:
        raise SourceArtifactError(
            "Non-fixture source artifacts must carry source_manifest_hash for compile provenance."
        )
    if artifact_role in {"processed_events", "normalized_facts"} and row_count == 0:
        warnings.append("empty_processed_source_artifact")

    return SourceArtifact(
        source_system=source_system,
        artifact_role=artifact_role,
        path=str(artifact_path),
        content_hash=_sha256_file(artifact_path),
        byte_size=artifact_path.stat().st_size,
        row_count=row_count,
        columns=columns,
        provenance_mode=provenance_mode,
        source_manifest_hash=source_manifest_hash,
        manifest_path=str(manifest_path) if manifest_path is not None else None,
        warnings=warnings,
        inspected_at=_now(),
    )


def _compile_source_mode(artifacts: list[SourceArtifact]) -> str:
    modes = {artifact.provenance_mode for artifact in artifacts}
    if not modes:
        return "empty"
    if modes == {"fixture"}:
        return "fixture_only"
    if "fixture" in modes:
        return "mixed_fixture_external"
    return "materialized_external"


def write_source_artifact_manifest(
    *,
    artifacts: list[SourceArtifact],
    output_path: str | Path,
    manifest_id: str = "source_artifact_manifest",
    purpose: str = "compile_source_reality_gate",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "purpose": purpose,
        "created_at": _now(),
        "compile_source_mode": _compile_source_mode(artifacts),
        "artifact_count": len(artifacts),
        "artifacts": [artifact.as_manifest() for artifact in artifacts],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_source_artifact_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_source_artifact_manifest(
    *,
    manifest_path: str | Path,
    require_materialized_external: bool = False,
    required_roles: Iterable[str] | None = None,
    required_systems: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest = load_source_artifact_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = list(manifest.get("artifacts") or [])

    if manifest.get("schema_version") != SOURCE_ARTIFACT_SCHEMA_VERSION:
        errors.append("source artifact manifest schema_version mismatch")

    seen_roles = {str(item.get("artifact_role")) for item in artifacts}
    seen_systems = {str(item.get("source_system")) for item in artifacts}

    for role in _as_required_set(required_roles):
        if role not in seen_roles:
            errors.append(f"missing required artifact_role: {role}")
    for system in _as_required_set(required_systems):
        if system not in seen_systems:
            errors.append(f"missing required source_system: {system}")

    for item in artifacts:
        path = Path(str(item.get("path", "")))
        if not path.exists():
            errors.append(f"artifact path missing: {path}")
            continue
        actual_hash = _sha256_file(path)
        if actual_hash != item.get("content_hash"):
            errors.append(f"artifact hash mismatch: {path}")
        if item.get("provenance_mode") == "fixture":
            warnings.append(f"fixture artifact present: {item.get('source_system')}:{item.get('artifact_role')}")
        if item.get("provenance_mode") != "fixture" and not item.get("source_manifest_hash"):
            errors.append(f"non-fixture artifact missing source_manifest_hash: {path}")

    if require_materialized_external and manifest.get("compile_source_mode") != "materialized_external":
        errors.append(
            "source artifact manifest is not production-candidate materialized_external mode"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "manifest_path": str(manifest_path),
        "compile_source_mode": manifest.get("compile_source_mode"),
        "artifact_count": len(artifacts),
        "source_systems": sorted(seen_systems),
        "artifact_roles": sorted(seen_roles),
    }


def source_manifest_summary(*, manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_source_artifact_manifest(manifest_path)
    artifacts = list(manifest.get("artifacts") or [])
    return {
        "manifest_path": str(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "compile_source_mode": manifest.get("compile_source_mode"),
        "artifact_count": len(artifacts),
        "artifacts": [
            {
                "source_system": item.get("source_system"),
                "artifact_role": item.get("artifact_role"),
                "provenance_mode": item.get("provenance_mode"),
                "row_count": item.get("row_count"),
                "byte_size": item.get("byte_size"),
                "path": item.get("path"),
            }
            for item in artifacts
        ],
    }
