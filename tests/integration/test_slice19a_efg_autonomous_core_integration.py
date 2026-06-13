from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.efg.promotion_plan import materialized_fields_from_manifest
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest
from pegasus.workflows.build_efg import build_autonomous_efg_from_manifest


def test_slice19a_autonomous_workflow_profiles_real_artifact_and_writes_handoff(tmp_path: Path) -> None:
    artifact = tmp_path / "sim_processed.parquet"
    pl.DataFrame({
        "year": [2021, 2022, 2022],
        "mun_residence_cod6": ["270430", "270430", "270430"],
        "underlying_icd_norm": ["A00", "B20", "A00"],
        "event_id": ["a", "b", "c"],
        "constant_unknown": [1, 1, 1],
        "all_missing_unknown": [None, None, None],
    }).write_parquet(artifact)
    substrate = build_substrate_bundle(artifacts=[SourceArtifactRef(
        path=str(artifact),
        source_system="SIM-DO",
        provenance_mode="fixture",
        source_manifest_hash="integration-manifest",
    )])
    substrate_path = write_substrate_bundle_manifest(substrate, tmp_path / "substrate.json")
    output_path = tmp_path / "Tables" / "efg_autonomous_manifest.json"

    result = build_autonomous_efg_from_manifest(
        substrate_manifest=substrate_path,
        output_path=output_path,
        operator_mode="standard",
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["efg_id"] == result.efg_id
    assert any(field.name.endswith(".count") for field in result.fields)
    assert any(field.unit == "ICD10" for field in result.fields)
    assert not any(field.support.get("column") == "event_id" for field in result.fields)
    assert not any(field.support.get("column") == "constant_unknown" for field in result.fields)
    assert result.edges
    assert materialized_fields_from_manifest(payload)
    assert set(payload) >= {"fields", "edges", "failed_branches", "precompression"}
