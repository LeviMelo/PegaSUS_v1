from __future__ import annotations

import json
from pathlib import Path

import pytest

from pegasus.acceptance.contracts import evaluate_level3_acceptance, summarize_run
from pegasus.output.table_io import read_rows
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest
from pegasus.source_artifacts.compile_policy import CompileSourceRealityError
from pegasus.workflows.compile import _write_sidra_smoke_facts, run_compile
from pegasus.workflows.datasus import run_datasus_normalize_sim
from pegasus.workflows.sinasc import run_datasus_normalize_sinasc


def test_slice12b_compile_records_missing_manifest_as_fixture_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_fixture_only"
    result = run_compile(
        intent_path=Path("config/intents/alagoas_smoke.json"),
        run_dir=run_dir,
    )
    assert result["validation"].ok is True

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    user_intent = json.loads((run_dir / "UserIntent.json").read_text(encoding="utf-8"))
    repro = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

    assert run_config["compile_source_mode"] == "fixture_only"
    assert run_config["source_artifact_manifest_present"] is False
    assert run_config["source_reality_production_candidate"] is False
    assert p_vector["source_artifact_reality"]["compile_source_mode"] == "fixture_only"
    assert user_intent["compile_source_reality"]["compile_source_mode"] == "fixture_only"
    assert repro["compile_source_mode"] == "fixture_only"

    summary = summarize_run(run_dir)
    manifest = summary.as_manifest()
    assert manifest["compile_source_mode"] == "fixture_only"
    assert manifest["source_artifact_manifest_present"] is False
    fixture_text = " ".join(
        str(value)
        for row in read_rows(run_dir / "V_fields.parquet")
        for value in row.values()
    ).lower()
    assert "fixture" in fixture_text


def test_slice12b_compile_strict_mode_aborts_without_manifest(tmp_path: Path) -> None:
    with pytest.raises(CompileSourceRealityError):
        run_compile(
            intent_path=Path("config/intents/alagoas_smoke.json"),
            run_dir=tmp_path / "strict_compile",
            require_materialized_external=True,
        )


def test_strict_compile_uses_materialized_external_inputs_and_reaches_production_candidate(tmp_path: Path) -> None:
    sim_path = tmp_path / "sim_events.parquet"
    sinasc_path = tmp_path / "sinasc_events.parquet"
    sidra_path = tmp_path / "sidra_facts.parquet"
    run_datasus_normalize_sim(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv", output_path=sim_path,
        source_manifest_hash="external_sim_request",
    )
    run_datasus_normalize_sinasc(
        input_path="tests/fixtures/datasus/sinasc_fixture.csv", output_path=sinasc_path,
        source_manifest_hash="external_sinasc_request",
    )
    _write_sidra_smoke_facts(output_path=sidra_path)
    artifacts = [
        inspect_source_artifact(path=sim_path, source_system="SIM-DO", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash="external_sim_request"),
        inspect_source_artifact(path=sinasc_path, source_system="SINASC", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash="external_sinasc_request"),
        inspect_source_artifact(path=sidra_path, source_system="SIDRA", artifact_role="normalized_facts", provenance_mode="materialized_external", source_manifest_hash="external_sidra_request"),
    ]
    source_manifest = write_source_artifact_manifest(artifacts=artifacts, output_path=tmp_path / "source_manifest.json")
    run_dir = tmp_path / "production_run"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json", run_dir=run_dir,
        data_root=tmp_path / "data", source_manifest=source_manifest,
        require_materialized_external=True,
    )
    assert result["validation"].ok
    acceptance = evaluate_level3_acceptance(run_dir)
    assert acceptance.ok, acceptance.errors
    assert acceptance.status == "production_candidate"
    assert acceptance.production_candidate is True

    for name in ("V_fields", "Warnings", "VariableDictionary", "E_DAG"):
        rows = read_rows(run_dir / f"{name}.parquet")
        table_text = " ".join(str(value) for row in rows for value in row.values()).lower()
        assert "fixture" not in table_text, name
        assert "synthetic" not in table_text, name

    field_rows = read_rows(run_dir / "V_fields.parquet")
    field_names = {str(row["name"]) for row in field_rows}
    assert "FixturePopulation" not in field_names
    assert "SIMCrudeMortalityFixture" not in field_names
    assert "IBGESelfDeclaredPopulationPlaceholder" not in field_names
