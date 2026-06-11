from pathlib import Path
import json

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.sidra.facts import normalize_fixture_json_to_facts
from pegasus.workflows.population import run_population_tensor_fixture, run_population_tensor_plan


def _facts(tmp_path: Path) -> Path:
    out = tmp_path / "sidra_population.parquet"
    normalize_fixture_json_to_facts(
        input_path="tests/fixtures/sidra/sidra_9606_population_tensor_fixture.json",
        output_path=out,
        table_id="9606",
        unit_by_variable={"93": "Pessoas"},
    )
    return out


def test_slice6a_population_tensor_bundle_validates_and_records_metadata(tmp_path: Path):
    facts = _facts(tmp_path)
    run = tmp_path / "population_run"
    result = run_population_tensor_fixture(sidra_facts_path=facts, run_dir=run, mode="independent_denominator")
    assert result["validation"].ok, result["validation"].errors
    assert validate_output_bundle(run_dir=str(run)).ok

    v = pl.read_parquet(run / "V_fields.parquet")
    names = set(v["name"].to_list())
    assert "PopulationTensorOfficialSIDRAIndependent" in names
    field = v.to_dicts()[0]
    support = json.loads(field["support_json"])
    assert support["PopulationTensorMode"] == "independent_denominator"
    assert support["SolverBackend"] == "algebraic_sidra_anchor_identity"
    assert support["SparseJacobian"] is True
    assert support["DenominatorFeedbackWarning"] is False
    assert support["reconstruction_uncertainty"] == 0.0

    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert manifest["population_tensor"]["PopulationTensorMode"] == "independent_denominator"
    assert manifest["population_tensor"]["SolverBackend"] == "algebraic_sidra_anchor_identity"
    assert manifest["telemetry"]["stage_status"]["population_solver"] == "success"

    failed = set(pl.read_parquet(run / "FailedBranches.parquet")["failed_branch_id"].to_list())
    assert "failed_dense_national_population_tensor_above_threshold" in failed


def test_slice6a_population_tensor_plan_exposes_sim_feedback_warning(tmp_path: Path):
    plan = run_population_tensor_plan(sidra_facts_path=_facts(tmp_path), mode="sim_informed_denominator")
    assert plan["DenominatorFeedbackWarning"] is True
    assert "sim_informed_population_feedback_risk" in plan["warnings"]
