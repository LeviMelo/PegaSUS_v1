"""Workflow wrappers for Slice 12A source artifact reality gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.source_artifacts.contracts import (
    inspect_source_artifact,
    source_manifest_summary,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)


def run_source_artifact_inspect(
    *,
    path: str | Path,
    source_system: str,
    artifact_role: str,
    provenance_mode: str,
    output: str | Path | None = None,
    source_manifest_hash: str | None = None,
) -> dict[str, Any]:
    artifact = inspect_source_artifact(
        path=path,
        source_system=source_system,
        artifact_role=artifact_role,
        provenance_mode=provenance_mode,
        source_manifest_hash=source_manifest_hash,
    )
    if output is not None:
        write_source_artifact_manifest(artifacts=[artifact], output_path=output)
    return artifact.as_manifest()


def run_source_manifest_validate(
    *,
    manifest: str | Path,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    return validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=require_materialized_external,
    )


def run_source_manifest_summary(*, manifest: str | Path) -> dict[str, Any]:
    return source_manifest_summary(manifest_path=manifest)
