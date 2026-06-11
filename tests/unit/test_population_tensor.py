from pathlib import Path

import pytest

from pegasus.registries.population import PopulationTensorScaleError, get_population_solver, select_population_solver
from pegasus.she.population.solvers import dense_national_abort_check, solve_population_tensor_from_sidra_anchor
from pegasus.sidra.facts import normalize_fixture_json_to_facts


def _facts(tmp_path: Path) -> Path:
    out = tmp_path / "sidra_population.parquet"
    normalize_fixture_json_to_facts(
        input_path="tests/fixtures/sidra/sidra_9606_population_tensor_fixture.json",
        output_path=out,
        table_id="9606",
        unit_by_variable={"93": "Pessoas"},
    )
    return out


def test_population_solver_registry_selects_independent_mode():
    spec = select_population_solver(mode="independent_denominator")
    assert spec.solver_id == "independent_sidra_anchor_v1"
    assert get_population_solver(spec.solver_id).sparse_jacobian is True


def test_independent_population_tensor_solves_from_sidra_anchor(tmp_path: Path):
    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=_facts(tmp_path), mode="independent_denominator")
    assert result.mode == "independent_denominator"
    assert result.value == 1025360.0
    assert result.denominator_feedback_warning is False
    assert result.diagnostics.sparse_jacobian is True
    assert result.reconstruction_uncertainty == 0.0


def test_sim_informed_population_tensor_emits_feedback_warning(tmp_path: Path):
    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=_facts(tmp_path), mode="sim_informed_denominator")
    assert result.denominator_feedback_warning is True
    assert "sim_informed_population_feedback_risk" in result.warnings
    assert result.state == "fragile"


def test_dense_national_population_tensor_aborts_above_threshold():
    with pytest.raises(PopulationTensorScaleError):
        dense_national_abort_check(localities=10_000, periods=20, strata=10)
