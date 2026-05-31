from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pegasus.core.hashing import sha256_file, sha256_json
from pegasus.core.manifests import utc_now_iso


class DatasusLocalWorkflowManifest(BaseModel):
    workflow_id: str
    source_system: str
    status: Literal["success", "failed"]

    created_at: str

    raw_path: str
    raw_sha256: str
    processed_path: str | None = None
    processed_sha256: str | None = None

    normalization_input_kind: Literal["raw", "processed"]
    normalizer: str

    raw_profile_path: str
    processed_profile_path: str | None = None
    schema_comparison_path: str | None = None
    normalized_path: str

    n_raw_rows: int
    n_processed_rows: int | None = None
    n_normalized_rows: int

    warnings: list[str] = Field(default_factory=list)


def build_local_workflow_id(
    *,
    source_system: str,
    raw_path: str | Path,
    processed_path: str | Path | None,
    raw_sha256: str,
    processed_sha256: str | None,
    normalization_input_kind: str,
) -> str:
    payload = {
        "source_system": source_system,
        "raw_path": str(Path(raw_path)),
        "processed_path": None if processed_path is None else str(Path(processed_path)),
        "raw_sha256": raw_sha256,
        "processed_sha256": processed_sha256,
        "normalization_input_kind": normalization_input_kind,
    }
    return sha256_json(payload)


def build_success_manifest(
    *,
    source_system: str,
    raw_path: str | Path,
    processed_path: str | Path | None,
    normalization_input_kind: Literal["raw", "processed"],
    normalizer: str,
    raw_profile_path: str | Path,
    processed_profile_path: str | Path | None,
    schema_comparison_path: str | Path | None,
    normalized_path: str | Path,
    n_raw_rows: int,
    n_processed_rows: int | None,
    n_normalized_rows: int,
    warnings: list[str] | None = None,
) -> DatasusLocalWorkflowManifest:
    raw_sha256 = sha256_file(raw_path)
    processed_sha256 = sha256_file(processed_path) if processed_path is not None else None

    workflow_id = build_local_workflow_id(
        source_system=source_system,
        raw_path=raw_path,
        processed_path=processed_path,
        raw_sha256=raw_sha256,
        processed_sha256=processed_sha256,
        normalization_input_kind=normalization_input_kind,
    )

    return DatasusLocalWorkflowManifest(
        workflow_id=workflow_id,
        source_system=source_system,
        status="success",
        created_at=utc_now_iso(),
        raw_path=str(raw_path),
        raw_sha256=raw_sha256,
        processed_path=None if processed_path is None else str(processed_path),
        processed_sha256=processed_sha256,
        normalization_input_kind=normalization_input_kind,
        normalizer=normalizer,
        raw_profile_path=str(raw_profile_path),
        processed_profile_path=None
        if processed_profile_path is None
        else str(processed_profile_path),
        schema_comparison_path=None
        if schema_comparison_path is None
        else str(schema_comparison_path),
        normalized_path=str(normalized_path),
        n_raw_rows=n_raw_rows,
        n_processed_rows=n_processed_rows,
        n_normalized_rows=n_normalized_rows,
        warnings=warnings or [],
    )