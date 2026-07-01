"""SIDRA plan, extract, normalize, and source-artifact workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.sidra.api import SidraClient
from pegasus.sidra.extract import extract_chunk_plan, write_chunk_plan, write_extraction_log
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.schemas import SIDRAMetadata, SIDRARequest
from pegasus.source_artifacts.contracts import inspect_source_artifact, validate_source_artifact_manifest, write_source_artifact_manifest


def ingest_sidra(
    *,
    request: SIDRARequest,
    metadata: SIDRAMetadata,
    output_manifest: str | Path,
    work_dir: str | Path,
    client: SidraClient | None = None,
    concurrency: int = 4,
    max_cells_per_request: int = 49_900,
) -> dict[str, Any]:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=max_cells_per_request)
    plan_path = write_chunk_plan(chunks, output_path=root / "chunk_plan.json")
    table_metadata = metadata.tables[request.table_id]
    metadata_hash = content_hash(table_metadata.model_dump(mode="json"))
    results = extract_chunk_plan(
        chunks, client=client or SidraClient(), concurrency=concurrency,
        raw_dir=root / "raw", facts_root=root / "facts", metadata_hash=metadata_hash,
        unit_by_variable=table_metadata.units_by_variable,
    )
    log_path = write_extraction_log(results, output_path=root / "extraction_log.json")
    if any(item.status != "success" for item in results):
        return {"status": "failed", "ok": False, "plan_path": str(plan_path), "extraction_log": str(log_path), "results": [item.model_dump(mode="json") for item in results]}
    extraction_hash = sha256_file(log_path)
    artifacts = [
        inspect_source_artifact(
            path=item.raw_path, source_system="SIDRA", artifact_role="raw_payload",
            provenance_mode="materialized_external", source_manifest_hash=extraction_hash, manifest_path=log_path,
        )
        for item in results if item.raw_path
    ]
    artifacts.extend(
        inspect_source_artifact(
            path=item.facts_path, source_system="SIDRA", artifact_role="normalized_facts",
            provenance_mode="materialized_external", source_manifest_hash=extraction_hash, manifest_path=log_path,
        )
        for item in results if item.facts_path
    )
    manifest_path = write_source_artifact_manifest(artifacts=artifacts, output_path=output_manifest, manifest_id="sidra_materialized_external")
    validation = validate_source_artifact_manifest(manifest_path=manifest_path, require_materialized_external=True, required_systems=("SIDRA",), required_roles=("normalized_facts",))
    return {"status": "materialized" if validation["ok"] else "failed", "ok": validation["ok"], "plan_path": str(plan_path), "extraction_log": str(log_path), "source_manifest": str(manifest_path), "validation": validation, "results": [item.model_dump(mode="json") for item in results]}
