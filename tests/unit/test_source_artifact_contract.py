from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pegasus.source_artifacts.contracts import (
    SourceArtifactError,
    inspect_source_artifact,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)


def test_source_artifact_inspection_marks_fixture_not_production_candidate(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "municipality_cod6": ["270430"], "year": [2022]}).write_parquet(path)

    artifact = inspect_source_artifact(
        path=path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
        required_columns=["event_id", "year"],
    )

    assert artifact.row_count == 1
    assert "event_id" in artifact.columns
    assert artifact.provenance_mode == "fixture"
    assert "fixture_source_artifact_not_production_candidate" in artifact.warnings


def test_non_fixture_artifacts_require_source_manifest_hash(tmp_path: Path) -> None:
    path = tmp_path / "sidra_facts.parquet"
    pl.DataFrame({"table_id": ["9606"], "value_numeric": [1.0]}).write_parquet(path)

    with pytest.raises(SourceArtifactError, match="source_manifest_hash"):
        inspect_source_artifact(
            path=path,
            source_system="SIDRA",
            artifact_role="normalized_facts",
            provenance_mode="materialized_external",
        )


def test_manifest_validation_blocks_fixture_when_external_required(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2022]}).write_parquet(path)
    artifact = inspect_source_artifact(
        path=path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
    )
    manifest = tmp_path / "source_manifest.json"
    write_source_artifact_manifest(artifacts=[artifact], output_path=manifest)

    relaxed = validate_source_artifact_manifest(manifest_path=manifest)
    strict = validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=True,
    )

    assert relaxed["ok"] is True
    assert strict["ok"] is False
    assert any("materialized_external" in error for error in strict["errors"])


def test_materialized_external_manifest_is_production_candidate(tmp_path: Path) -> None:
    path = tmp_path / "sidra_facts.parquet"
    pl.DataFrame({"table_id": ["9606"], "value_numeric": [1.0]}).write_parquet(path)
    artifact = inspect_source_artifact(
        path=path,
        source_system="SIDRA",
        artifact_role="normalized_facts",
        provenance_mode="materialized_external",
        source_manifest_hash="sidra_request_manifest_hash",
    )
    manifest = tmp_path / "source_manifest.json"
    write_source_artifact_manifest(artifacts=[artifact], output_path=manifest)

    result = validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=True,
        required_roles=["normalized_facts"],
        required_systems=["SIDRA"],
    )

    assert result["ok"] is True
    assert result["compile_source_mode"] == "materialized_external"
