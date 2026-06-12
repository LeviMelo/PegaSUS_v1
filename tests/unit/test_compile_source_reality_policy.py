from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pegasus.source_artifacts.compile_policy import (
    CompileSourceRealityError,
    resolve_compile_source_reality,
)
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest


def test_slice12b_missing_source_manifest_is_explicit_fixture_only() -> None:
    reality = resolve_compile_source_reality()
    assert reality.compile_source_mode == "fixture_only"
    assert reality.source_artifact_manifest_present is False
    assert reality.production_candidate is False
    assert "compile_source_manifest_missing_assumed_fixture_only" in reality.validation_warnings


def test_slice12b_strict_compile_requires_manifest() -> None:
    with pytest.raises(CompileSourceRealityError):
        resolve_compile_source_reality(require_materialized_external=True)


def test_slice12b_materialized_external_manifest_is_production_candidate(tmp_path: Path) -> None:
    artifact_path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2020]}).write_parquet(artifact_path)
    artifact = inspect_source_artifact(
        path=artifact_path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="materialized_external",
        source_manifest_hash="datasus_manifest_hash",
    )
    manifest_path = tmp_path / "source_artifacts.json"
    write_source_artifact_manifest(artifacts=[artifact], output_path=manifest_path)

    reality = resolve_compile_source_reality(
        source_manifest=manifest_path,
        require_materialized_external=True,
    )

    assert reality.compile_source_mode == "materialized_external"
    assert reality.source_artifact_manifest_present is True
    assert reality.production_candidate is True
    assert reality.source_artifact_count == 1
    assert reality.source_systems == ("SIM-DO",)
