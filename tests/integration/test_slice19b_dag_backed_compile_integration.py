from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


def test_slice19b_compile_routes_normalized_sources_through_autonomous_efg(tmp_path: Path) -> None:
    run = tmp_path / "compile_19b"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run,
        data_root=tmp_path / "data",
    )

    assert result["validation"].ok, result["validation"].errors
    assert validate_output_bundle(run_dir=str(run)).ok
    assert set(path.name for path in run.iterdir()) == set(OUTPUT_BUNDLE_FILES.values())

    metadata = result["autonomous_efg"]
    assert metadata["status"] == "attached"
    assert metadata["graph_authority"] == "autonomous_efg_core"
    assert metadata["legacy_computed_fields_preserved"] is True
    assert metadata["first_class_output_keys_added"] == 0
    assert metadata["source_systems"] == ["SIDRA", "SIM-DO", "SINASC"]
    assert metadata["field_count"] > 0
    assert metadata["edge_count"] > 0

    manifest_path = run / metadata["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["efg_id"] == metadata["efg_id"]
    assert manifest["precompression"]["input_count"] >= manifest["precompression"]["output_count"]
    assert manifest["failed_branches"]

    fields = pq.read_table(run / "V_fields.parquet").to_pylist()
    names = {str(row["name"]) for row in fields}
    assert "SIMDeathsAll" in names
    assert "SIMCrudeMortalitySIDRAOfficial" in names
    autonomous_ids = {
        str(item["field"]["id"])
        for item in manifest["fields"]
    }
    bundle_ids = {str(row["field_id"]) for row in fields}
    assert autonomous_ids <= bundle_ids
    assert all(
        row["state"] == "quarantined_descriptive"
        for row in fields
        if row["field_id"] in autonomous_ids
    )

    edges = pq.read_table(run / "E_DAG.parquet").to_pylist()
    assert any(edge["child_field_id"] in autonomous_ids for edge in edges)
    assert all(edge["parent_field_id"] in bundle_ids for edge in edges)
    assert all(edge["child_field_id"] in bundle_ids for edge in edges)

    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
    reproducibility = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert run_config["autonomous_efg"]["efg_id"] == metadata["efg_id"]
    assert reproducibility["autonomous_efg"]["manifest_hash"] == metadata["manifest_hash"]
    assert reproducibility["source_hashes"]["autonomous_efg_manifest"] == metadata["manifest_hash"]
