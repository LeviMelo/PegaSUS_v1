
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.efg.materialization_manifest import attach_efg_materialization_summary_to_run
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest


def test_slice14b_attaches_efg_materialization_manifest_without_new_first_class_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifact = tmp_path / "sim.parquet"
    create_empty_output_bundle(run_dir)

    pl.DataFrame(
        {
            "year": [2020, 2021, 2021],
            "age_years": [50, 51, 52],
            "race_color_admin": ["1", "4", ""],
            "underlying_icd_norm": ["I10", "J18", "R99"],
            "constant_col": [1, 1, 1],
            "all_missing_col": [None, None, None],
        }
    ).write_parquet(artifact)

    bundle = build_substrate_bundle(
        artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
    )
    substrate_manifest = run_dir / "Tables" / "substrate_manifest.json"
    write_substrate_bundle_manifest(bundle, substrate_manifest)

    before_keys = sorted(path.name for path in run_dir.iterdir())
    summary = attach_efg_materialization_summary_to_run(run_dir=run_dir)
    after_keys = sorted(path.name for path in run_dir.iterdir())

    assert before_keys == after_keys
    assert summary["status"] == "evaluated"
    assert summary["metadata_only"] is True
    assert summary["writes_v_fields"] is False
    assert summary["writes_e_dag"] is False
    assert summary["field_count"] == len(bundle.candidates)
    assert summary["excluded_field_count"] == len(bundle.exclusions)
    assert "constant_col" in summary["excluded_columns"]
    assert "all_missing_col" in summary["excluded_columns"]
    assert (run_dir / "Tables" / "efg_substrate_materialization.json").exists()

    manifest = json.loads((run_dir / "Tables" / "efg_substrate_materialization.json").read_text(encoding="utf-8"))
    materialized_columns = {
        item["field"]["support"]["column"]
        for item in manifest["fields"]
    }
    excluded_columns = {item["column"] for item in manifest["excluded_source_fields"]}
    assert materialized_columns.isdisjoint(excluded_columns)
    assert any(item["field"]["unit"] == "ICD10" and item["field"]["kind"] == "observer_proxy" for item in manifest["fields"])

    for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        payload = json.loads((run_dir / name).read_text(encoding="utf-8"))
        assert payload["efg_materialization_gate"]["status"] == "evaluated"
        assert payload["efg_materialization_gate"]["manifest_path"] == "Tables/efg_substrate_materialization.json"

    assert validate_output_bundle(run_dir=str(run_dir)).ok
