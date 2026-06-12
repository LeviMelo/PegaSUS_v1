from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.build_substrate import (
    run_attach_substrate_to_run,
    run_build_substrate_from_artifacts,
    run_substrate_summary,
)


def test_slice13a_substrate_workflow_writes_manifest_and_summary(tmp_path: Path) -> None:
    events = tmp_path / "sim_events.parquet"
    out = tmp_path / "substrate.json"
    pl.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "age_years": [20.0, 21.0, 22.0, 23.0],
        "constant_cost": [0, 0, 0, 0],
        "all_missing_cost": [None, None, None, None],
        "underlying_icd_norm": ["I10", "I10", "J18", "J18"],
    }).write_parquet(events)

    payload = run_build_substrate_from_artifacts(
        artifacts=[{
            "path": str(events),
            "source_system": "SIM-DO",
            "artifact_role": "processed_events",
            "provenance_mode": "fixture",
            "source_manifest_hash": "fixture_manifest_hash",
        }],
        output=out,
    )

    assert out.exists()
    assert payload["source_reality_mode"] == "fixture_only"
    assert payload["admissible_candidate_count"] >= 2
    assert payload["zero_variance_exclusion_count"] >= 1
    assert payload["all_missing_exclusion_count"] == 1
    summary = run_substrate_summary(manifest=out)
    assert summary["substrate_id"] == payload["substrate_id"]


def test_slice13a_attach_substrate_metadata_to_valid_run_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    events = tmp_path / "sih_events.parquet"
    create_empty_output_bundle(run_dir)
    pl.DataFrame({
        "admission_id": ["x", "y", "z"],
        "principal_icd_norm": ["G43", "G43", "I10"],
        "stay_length_days": [1, 2, 3],
        "total_admission_cost_real": [100.0, 120.0, 150.0],
        "zero_val_sadt": [0, 0, 0],
    }).write_parquet(events)

    summary = run_attach_substrate_to_run(
        run_dir=run_dir,
        artifacts=[{
            "path": str(events),
            "source_system": "SIH-RD",
            "artifact_role": "processed_events",
            "provenance_mode": "fixture",
            "source_manifest_hash": "fixture_manifest_hash",
        }],
    )

    assert summary["status"] == "evaluated"
    assert summary["admissible_candidate_count"] >= 3
    assert summary["zero_variance_exclusion_count"] >= 1
    assert (run_dir / "Tables" / "substrate_manifest.json").exists()
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["substrate_gate"]["substrate_id"] == summary["substrate_id"]
