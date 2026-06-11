"""Source artifact resolver helpers for manifest-driven compiler inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    inspect_source_artifact,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)


def build_manifest_from_paths(
    *,
    output_path: str | Path,
    entries: Iterable[dict],
    manifest_id: str = "source_artifact_manifest",
) -> Path:
    artifacts: list[SourceArtifact] = []
    for entry in entries:
        artifacts.append(
            inspect_source_artifact(
                path=entry["path"],
                source_system=entry["source_system"],
                artifact_role=entry["artifact_role"],
                provenance_mode=entry.get("provenance_mode", "fixture"),
                source_manifest_hash=entry.get("source_manifest_hash"),
                manifest_path=entry.get("manifest_path"),
                required_columns=entry.get("required_columns") or [],
            )
        )
    return write_source_artifact_manifest(
        artifacts=artifacts,
        output_path=output_path,
        manifest_id=manifest_id,
    )


def compile_reality_gate(
    *,
    manifest_path: str | Path,
    require_materialized_external: bool = False,
) -> dict:
    return validate_source_artifact_manifest(
        manifest_path=manifest_path,
        require_materialized_external=require_materialized_external,
        required_roles=["processed_events", "normalized_facts"],
    )
