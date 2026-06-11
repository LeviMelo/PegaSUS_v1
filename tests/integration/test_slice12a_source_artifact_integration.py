from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.source_artifacts.contracts import (
    inspect_source_artifact,
    source_manifest_summary,
    validate_source_artifact_manifest,
    write_source_artifact_manifest,
)
from pegasus.workflows.source_artifacts import (
    run_source_artifact_inspect,
    run_source_manifest_summary,
    run_source_manifest_validate,
)


def test_slice12a_source_manifest_distinguishes_fixture_from_external(tmp_path: Path) -> None:
    sim_path = tmp_path / "sim_events.parquet"
    sidra_path = tmp_path / "sidra_facts.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2022]}).write_parquet(sim_path)
    pl.DataFrame({"table_id": ["9606"], "value_numeric": [123.0]}).write_parquet(sidra_path)

    artifacts = [
        inspect_source_artifact(
            path=sim_path,
            source_system="SIM-DO",
            artifact_role="processed_events",
            provenance_mode="fixture",
        ),
        inspect_source_artifact(
            path=sidra_path,
            source_system="SIDRA",
            artifact_role="normalized_facts",
            provenance_mode="materialized_external",
            source_manifest_hash="sidra_manifest_hash",
        ),
    ]
    manifest = tmp_path / "source_artifacts.json"
    write_source_artifact_manifest(artifacts=artifacts, output_path=manifest)

    result = validate_source_artifact_manifest(manifest_path=manifest)
    strict = validate_source_artifact_manifest(
        manifest_path=manifest,
        require_materialized_external=True,
    )
    summary = source_manifest_summary(manifest_path=manifest)

    assert result["ok"] is True
    assert result["compile_source_mode"] == "mixed_fixture_external"
    assert strict["ok"] is False
    assert summary["artifact_count"] == 2


def test_slice12a_workflow_wrappers_are_json_ready(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({"event_id": ["a"], "year": [2022]}).write_parquet(path)
    manifest = tmp_path / "manifest.json"

    inspected = run_source_artifact_inspect(
        path=path,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
        output=manifest,
    )
    validation = run_source_manifest_validate(manifest=manifest)
    summary = run_source_manifest_summary(manifest=manifest)

    assert inspected["source_system"] == "SIM-DO"
    assert validation["ok"] is True
    assert summary["compile_source_mode"] == "fixture_only"
