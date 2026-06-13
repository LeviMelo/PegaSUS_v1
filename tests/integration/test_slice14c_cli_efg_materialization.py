
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from pegasus.cli import app
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, write_substrate_bundle_manifest


def _build_run_with_substrate(tmp_path: Path) -> tuple[Path, Path]:
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
    return run_dir, substrate_manifest


def test_slice14c_cli_materializes_inspects_and_attaches_without_new_first_class_key(tmp_path: Path) -> None:
    runner = CliRunner()
    run_dir, substrate_manifest = _build_run_with_substrate(tmp_path)
    output = tmp_path / "efg_materialization.json"

    materialize_result = runner.invoke(
        app,
        [
            "efg",
            "materialize-substrate-manifest",
            "--substrate-manifest",
            str(substrate_manifest),
            "--output",
            str(output),
        ],
    )
    assert materialize_result.exit_code == 0, materialize_result.output
    assert output.exists()
    materialized = json.loads(output.read_text(encoding="utf-8"))
    assert materialized["metadata_only"] is True
    assert materialized["writes_v_fields"] is False
    assert materialized["writes_e_dag"] is False
    assert any(item["field"]["unit"] == "ICD10" and item["field"]["kind"] == "observer_proxy" for item in materialized["fields"])

    inspect_result = runner.invoke(
        app,
        ["efg", "inspect-materialization", "--manifest", str(output)],
    )
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "field_count" in inspect_result.output
    assert "writes_v_fields" in inspect_result.output

    before_keys = sorted(path.name for path in run_dir.iterdir())
    attach_result = runner.invoke(
        app,
        ["efg", "attach-materialization", "--run-dir", str(run_dir)],
    )
    after_keys = sorted(path.name for path in run_dir.iterdir())
    assert attach_result.exit_code == 0, attach_result.output
    assert before_keys == after_keys
    assert (run_dir / "Tables" / "efg_substrate_materialization.json").exists()
    assert validate_output_bundle(run_dir=str(run_dir)).ok

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["efg_materialization_gate"]["status"] == "evaluated"
    assert run_config["efg_materialization_gate"]["writes_v_fields"] is False
