
from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest
from pegasus.workflows import compile as compile_workflow


def test_slice13c_run_compile_attaches_substrate_without_duplicate_public_api(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifact = tmp_path / "sim.parquet"
    manifest_path = tmp_path / "source_manifest.json"
    create_empty_output_bundle(run_dir)

    pl.DataFrame(
        {
            "year": [2020, 2021, 2021],
            "age_years": [50, 51, 52],
            "race_color_admin": ["1", "4", ""],
            "underlying_icd_norm": ["I10", "J18", "R99"],
            "constant_marker": [1, 1, 1],
            "all_missing_marker": [None, None, None],
        }
    ).write_parquet(artifact)

    source_artifact = inspect_source_artifact(
        path=artifact,
        source_system="SIM-DO",
        artifact_role="processed_events",
        provenance_mode="fixture",
    )
    write_source_artifact_manifest(artifacts=[source_artifact], output_path=manifest_path)

    def fake_impl(*, intent_path, run_dir=None, data_root="data", source_manifest=None, require_materialized_external=False):
        assert source_manifest == manifest_path
        return {
            "status": "success",
            "run_id": "fake",
            "run_dir": Path(run_dir),
            "validation": validate_output_bundle(run_dir=str(run_dir)),
        }

    monkeypatch.setattr(compile_workflow, "_run_compile_impl", fake_impl)

    result = compile_workflow.run_compile(
        intent_path=tmp_path / "intent.json",
        run_dir=run_dir,
        source_manifest=manifest_path,
    )

    assert result["substrate_gate"]["status"] == "evaluated"
    assert result["substrate_gate"]["admissible_candidate_count"] > 0
    assert result["substrate_gate"]["excluded_field_count"] > 0
    assert result["substrate_gate"]["source_reality_mode"] == "fixture_only"
    assert (run_dir / "Tables" / "substrate_manifest.json").exists()
    assert result["validation"].ok
