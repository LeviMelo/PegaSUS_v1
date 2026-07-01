from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.she.substrate import (
    SourceArtifactRef,
    attach_substrate_summary_to_run,
    build_substrate_bundle,
    load_source_artifacts_from_manifest,
    write_substrate_bundle_manifest,
)


def run_build_substrate_from_artifacts(
    *,
    artifacts: list[dict[str, Any]] | list[str],
    output: str | Path | None = None,
) -> dict[str, Any]:
    bundle = build_substrate_bundle(artifacts=artifacts)
    if output is not None:
        write_substrate_bundle_manifest(bundle, output)
    return bundle.as_manifest()


def run_build_substrate_from_source_manifest(
    *,
    source_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = load_source_artifacts_from_manifest(source_manifest)
    bundle = build_substrate_bundle(artifacts=artifacts)
    if output is not None:
        write_substrate_bundle_manifest(bundle, output)
    return bundle.as_manifest()


def run_attach_substrate_to_run(
    *,
    run_dir: str | Path,
    source_manifest: str | Path | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if source_manifest is not None:
        refs = load_source_artifacts_from_manifest(source_manifest)
    else:
        refs = [SourceArtifactRef(**artifact) for artifact in (artifacts or [])]
    bundle = build_substrate_bundle(artifacts=refs)
    return attach_substrate_summary_to_run(run_dir=run_dir, bundle=bundle)


def run_substrate_summary(*, manifest: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return {
        "schema_version": payload.get("schema_version"),
        "substrate_id": payload.get("substrate_id"),
        "source_reality_mode": payload.get("source_reality_mode"),
        "source_artifact_count": payload.get("source_artifact_count"),
        "admissible_candidate_count": payload.get("admissible_candidate_count"),
        "excluded_field_count": payload.get("excluded_field_count"),
        "zero_variance_exclusion_count": payload.get("zero_variance_exclusion_count"),
        "all_missing_exclusion_count": payload.get("all_missing_exclusion_count"),
        "structural_exclusion_count": payload.get("structural_exclusion_count"),
        "warnings": payload.get("warnings", []),
    }
