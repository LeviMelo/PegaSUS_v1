"""Source artifact reality contracts for DATASUS/SIDRA compiler inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import polars as pl


SOURCE_ARTIFACT_SCHEMA_VERSION = "2.0"

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
        separator = "\t" if suffix == ".tsv" else ","
        df = pl.read_csv(path, separator=separator)
        return df.height, list(df.columns)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return None, []
    return None, []


def inspect_source_artifact(
    *,
    path: str | Path,
    source_system: str,
    artifact_role: str,
    provenance_mode: str = "materialized_external",
    output: str | Path | None = None,
    source_manifest_hash: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(path)
    warnings: list[str] = []

    if source_system not in ALLOWED_SOURCE_SYSTEMS:
        raise SourceArtifactError(f"unsupported source_system: {source_system}")
    if artifact_role not in ALLOWED_ARTIFACT_ROLES:
        raise SourceArtifactError(f"unsupported artifact_role: {artifact_role}")
    if provenance_mode not in ALLOWED_PROVENANCE_MODES:
        raise SourceArtifactError(f"unsupported provenance_mode: {provenance_mode}")
    if not artifact_path.exists():
        raise SourceArtifactError(f"source artifact not found: {artifact_path}")

    row_count, columns = _table_shape(artifact_path)
    artifact = SourceArtifact(
        source_system=source_system,
        artifact_role=artifact_role,
        path=str(artifact_path),
        content_hash=_sha256_file(artifact_path),
        byte_size=artifact_path.stat().st_size,
        row_count=row_count,
        columns=columns,
        provenance_mode=provenance_mode,
        source_manifest_hash=source_manifest_hash,
        manifest_path=None if manifest_path is None else str(manifest_path),
        warnings=warnings,
        inspected_at=_now(),
    )
    payload = artifact.as_manifest()
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _artifact_payload(item: SourceArtifact | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, SourceArtifact):
        return item.as_manifest()
    return dict(item)


def write_source_artifact_manifest(
    artifacts: Iterable[SourceArtifact | dict[str, Any]],
    *,
    output: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = [_artifact_payload(item) for item in artifacts]
    modes = {str(row.get("provenance_mode")) for row in rows}
    if not rows:
        raise SourceArtifactError("source artifact manifest cannot be empty")
    if not modes <= ALLOWED_PROVENANCE_MODES:
        raise SourceArtifactError(f"unsupported provenance modes: {sorted(modes - ALLOWED_PROVENANCE_MODES)}")

    compile_source_mode = "materialized_external" if modes == {"materialized_external"} else "cached_external"
    payload = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "created_at": _now(),
        "compile_source_mode": compile_source_mode,
        "production_candidate": compile_source_mode == "materialized_external",
        "artifacts": rows,
        "source_systems": sorted({str(row.get("source_system")) for row in rows}),
        "artifact_roles": sorted({str(row.get("artifact_role")) for row in rows}),
    }
    out = Path(output if manifest_path is None else manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_source_artifact_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SourceArtifactError(f"source artifact manifest is not an object: {path}")
    return payload


def source_manifest_summary(manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
    payload = load_source_artifact_manifest(manifest) if not isinstance(manifest, dict) else dict(manifest)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []

    modes = {str(item.get("provenance_mode")) for item in artifacts if isinstance(item, dict)}
    source_systems = sorted({str(item.get("source_system")) for item in artifacts if isinstance(item, dict)})
    artifact_roles = sorted({str(item.get("artifact_role")) for item in artifacts if isinstance(item, dict)})

    if modes == {"materialized_external"}:
        compile_source_mode = "materialized_external"
    elif modes and modes <= ALLOWED_PROVENANCE_MODES:
        compile_source_mode = "cached_external"
    else:
        compile_source_mode = str(payload.get("compile_source_mode") or "invalid")

    return {
        "schema_version": payload.get("schema_version", SOURCE_ARTIFACT_SCHEMA_VERSION),
        "source_artifact_count": len(artifacts),
        "source_systems": source_systems,
        "artifact_roles": artifact_roles,
        "provenance_modes": sorted(modes),
        "compile_source_mode": compile_source_mode,
        "production_candidate": compile_source_mode == "materialized_external",
        "warnings": list(payload.get("warnings", []) or []),
    }


def validate_source_artifact_manifest(
    manifest: str | Path | dict[str, Any],
    *,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    payload = load_source_artifact_manifest(manifest) if not isinstance(manifest, dict) else dict(manifest)
    artifacts = payload.get("artifacts")
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(artifacts, list) or not artifacts:
        errors.append("source artifact manifest has no artifacts")
        artifacts = []

    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            errors.append(f"artifact {idx} is not an object")
            continue

        mode = str(item.get("provenance_mode"))
        source_system = str(item.get("source_system"))
        artifact_role = str(item.get("artifact_role"))
        path = Path(str(item.get("path", "")))

        if mode not in ALLOWED_PROVENANCE_MODES:
            errors.append(f"artifact {idx} has unsupported provenance_mode: {mode}")
        if source_system not in ALLOWED_SOURCE_SYSTEMS:
            errors.append(f"artifact {idx} has unsupported source_system: {source_system}")
        if artifact_role not in ALLOWED_ARTIFACT_ROLES:
            errors.append(f"artifact {idx} has unsupported artifact_role: {artifact_role}")
        if not path.exists():
            errors.append(f"artifact {idx} path does not exist: {path}")
        if mode == "materialized_external" and not item.get("content_hash"):
            errors.append(f"artifact {idx} materialized_external artifact missing content_hash")

    summary = source_manifest_summary(payload)
    if require_materialized_external and summary["compile_source_mode"] != "materialized_external":
        errors.append("source artifact manifest is not materialized_external")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
    }


__all__ = [
    "SourceArtifact",
    "SourceArtifactError",
    "inspect_source_artifact",
    "load_source_artifact_manifest",
    "source_manifest_summary",
    "validate_source_artifact_manifest",
    "write_source_artifact_manifest",
]
