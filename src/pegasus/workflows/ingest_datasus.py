"""DATASUS acquisition workflow and materialized-external source gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.core.hashing import sha256_file
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.source_artifacts.contracts import inspect_source_artifact, validate_source_artifact_manifest, write_source_artifact_manifest


def ingest_datasus(
    *,
    system: str,
    uf: str,
    years: str,
    output_manifest: str | Path,
    client: MicrodatasusClient | None = None,
) -> dict[str, Any]:
    result = (client or MicrodatasusClient()).fetch(system=system, uf=uf, years=years)
    if not result.ok:
        return {"status": "blocked", "ok": False, "batch": result.as_manifest(), "source_manifest": None}
    artifacts = []
    for request, request_manifest in zip(result.requests, result.manifest_paths, strict=True):
        request_hash = sha256_file(request_manifest)
        artifacts.extend((
            inspect_source_artifact(
                path=request.raw_path, source_system=request.system, artifact_role="raw_table",
                provenance_mode="materialized_external", source_manifest_hash=request_hash, manifest_path=request_manifest,
            ),
            inspect_source_artifact(
                path=request.processed_path, source_system=request.system, artifact_role="processed_events",
                provenance_mode="materialized_external", source_manifest_hash=request_hash, manifest_path=request_manifest,
            ),
        ))
    manifest_path = write_source_artifact_manifest(artifacts=artifacts, output_path=output_manifest, manifest_id="datasus_materialized_external")
    validation = validate_source_artifact_manifest(manifest_path=manifest_path, require_materialized_external=True)
    return {"status": "materialized" if validation["ok"] else "failed", "ok": validation["ok"], "batch": result.as_manifest(), "source_manifest": str(manifest_path), "validation": validation}
