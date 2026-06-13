from __future__ import annotations

import json
from pathlib import Path

from pegasus.acceptance.contracts import evaluate_level3_acceptance
from pegasus.workflows.compile import run_compile


def test_slice26a26b_compile_emits_authority_and_stage_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_compile(intent_path="config/intents/alagoas_smoke.json", run_dir=run_dir)
    assert result["validation"].ok
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    repro = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    architecture = run_config["compiler_architecture"]
    assert architecture["graph_authority"] == "autonomous_efg_core"
    assert architecture["numerical_materialization"] == "autonomous_compiler_services"
    assert architecture["legacy_bootstrap_status"] == "quarantined_fixture_only"
    assert architecture["legacy_graph_authority"] is False
    assert "compiler_stage_plan" in run_config
    assert "compiler_stage_plan" in repro
    assert run_config["compiler_stage_plan"]["by_stage"]["stdfm"]["skip_reason"]
    level3 = evaluate_level3_acceptance(run_dir)
    assert level3.ok, level3.as_manifest()
    assert level3.checks["compiler_stage_plan_present"] is True
    assert level3.checks["stage_skip_proofs_valid"] is True
