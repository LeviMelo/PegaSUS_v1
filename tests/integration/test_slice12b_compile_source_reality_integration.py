from __future__ import annotations

import json
from pathlib import Path

import pytest

from pegasus.acceptance.contracts import summarize_run
from pegasus.source_artifacts.compile_policy import CompileSourceRealityError
from pegasus.workflows.compile import run_compile


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


def test_slice12b_compile_strict_mode_aborts_without_manifest(tmp_path: Path) -> None:
    with pytest.raises(CompileSourceRealityError):
        run_compile(
            intent_path=Path("config/intents/alagoas_smoke.json"),
            run_dir=tmp_path / "strict_compile",
            require_materialized_external=True,
        )
