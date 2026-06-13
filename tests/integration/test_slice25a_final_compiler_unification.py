from __future__ import annotations

import json
from pathlib import Path

from pegasus.acceptance.contracts import evaluate_level3_acceptance
from pegasus.workflows.compile import run_compile


def test_slice25a_compiler_declares_autonomous_authority_and_level3_fixture_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "compile_25a"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
        data_root=tmp_path / "data",
    )

    assert result["validation"].ok, result["validation"].errors
    architecture = result["compiler_architecture"]
    assert architecture["graph_authority"] == "autonomous_efg_core"
    assert architecture["legacy_bootstrap_status"] == "compatibility_materializer"
    assert architecture["legacy_graph_authority"] is False

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert run_config["compiler_architecture"] == architecture
    assert manifest["compiler_architecture"] == architecture
    for stage in ("population_solver", "stdfm", "pirs_model", "pirs_hsic"):
        assert manifest["telemetry"]["stage_status"][stage] == "skipped"

    acceptance = evaluate_level3_acceptance(run_dir)
    assert acceptance.ok, acceptance.errors
    assert acceptance.status == "fixture_validated"
    assert acceptance.production_candidate is False
    assert all(acceptance.checks.values())
