from pathlib import Path

import pytest

from pegasus.core.exceptions import MemoryPreflightError
from pegasus.registries.population import (
    PopulationSolverUnavailableError,
    PopulationTensorScaleError,
    get_population_solver,
    select_population_solver,
)
from pegasus.she.population.schema import PopulationObjectiveWeights, PopulationTensorProblem
from pegasus.she.population.solvers import (
    dense_national_abort_check,
    solve_population_tensor_from_sidra_anchor,
    solve_population_tensor_problem,
)
from pegasus.she.population.sparse_admm import plan_sparse_population_solver, solve_sparse_population
from pegasus.she.population.state_space import build_population_state_space, solve_population_state_space_smoother
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
    assert spec.solver_id == "projected_gradient_small_v1"
    assert get_population_solver(spec.solver_id).sparse_jacobian is True


def test_population_solver_refuses_legacy_scaffold_backend():
    with pytest.raises(PopulationSolverUnavailableError, match="population_solver_unavailable_for_scale"):
        select_population_solver(
            mode="sim_informed_denominator",
            solver_id="sim_informed_sparse_admm_scaffold_v1",
        )


def test_independent_population_tensor_solves_from_sidra_anchor(tmp_path: Path):
    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=_facts(tmp_path), mode="independent_denominator")
    assert result.mode == "independent_denominator"
    assert result.value == 1025360.0
    assert result.denominator_feedback_warning is False
    assert result.diagnostics.sparse_jacobian is True
    assert result.reconstruction_uncertainty == 0.0
    assert result.solver_backend == "projected_gradient_small_sparse_analytic"
    assert result.diagnostics.converged is True
    assert result.diagnostics.final_objective == 0.0


def test_sim_informed_population_tensor_emits_feedback_warning(tmp_path: Path):
    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=_facts(tmp_path), mode="sim_informed_denominator")
    assert result.denominator_feedback_warning is True
    assert "sim_informed_population_feedback_risk" in result.warnings
    assert result.state == "fragile"


def test_dense_national_population_tensor_aborts_above_threshold():
    with pytest.raises(PopulationTensorScaleError):
        dense_national_abort_check(localities=10_000, periods=101, strata=10)


def test_population_optimizer_reconstructs_missing_cohort_cell_with_closure():
    problem = PopulationTensorProblem(
        shape=(1, 2, 3, 1, 1),
        anchors=(60.0, 30.0, 10.0, None, None, None),
        hard_anchor_mask=(True, True, True, False, False, False),
        closure_totals=(100.0, 150.0),
        births=(50.0, None),
        death_rates=(0.0,) * 6,
        migration_bounds=(0.0,) * 6,
        weights=PopulationObjectiveWeights(anchor=10.0, aging=2.0, birth=2.0, migration=0.1, age_smooth=0.0),
    )
    result = solve_population_tensor_problem(problem)
    assert result.telemetry.converged is True
    assert result.telemetry.final_objective < result.telemetry.initial_objective
    assert sum(result.population[3:]) == pytest.approx(150.0)
    assert result.population[3:] == pytest.approx((50.0, 60.0, 40.0), abs=0.1)


def test_sim_informed_loss_moves_population_toward_death_prior():
    problem = PopulationTensorProblem(
        shape=(1, 1, 1, 1, 1),
        anchors=(100.0,),
        mode="sim_informed_denominator",
        death_rates=(0.1,),
        sim_deaths=(20.0,),
        migration_bounds=(100.0,),
        weights=PopulationObjectiveWeights(anchor=1.0, death=10.0, aging=0.0, birth=0.0, migration=0.0, age_smooth=0.0),
    )
    result = solve_population_tensor_problem(problem)
    assert result.telemetry.converged is True
    assert result.population[0] > 100.0
    assert result.telemetry.objective_terms["death"] > 0.0


def test_population_optimizer_enforces_migration_enclosure():
    problem = PopulationTensorProblem(
        shape=(2, 1, 1, 1, 1),
        anchors=(10.0, 20.0),
        migration_bounds=(5.0, 5.0),
        migration_totals=(3.0,),
        initial_migration=(5.0, 5.0),
        weights=PopulationObjectiveWeights(aging=0.0, birth=0.0, migration=0.0, age_smooth=0.0),
    )
    result = solve_population_tensor_problem(problem)
    assert sum(result.migration) == pytest.approx(3.0)


def test_population_optimizer_applies_race_composition_prior():
    problem = PopulationTensorProblem(
        shape=(1, 1, 1, 1, 2),
        anchors=(None, None),
        closure_totals=(100.0,),
        initial_population=(90.0, 10.0),
        race_composition_prior=(0.5, 0.5),
        migration_bounds=(0.0, 0.0),
        weights=PopulationObjectiveWeights(anchor=0.0, aging=0.0, birth=0.0, migration=0.0, race=10.0, age_smooth=0.0),
    )
    result = solve_population_tensor_problem(problem)
    assert result.telemetry.final_objective < result.telemetry.initial_objective
    assert result.population == pytest.approx((50.0, 50.0), abs=0.1)


def test_sparse_population_matches_simple_projected_solution():
    problem = PopulationTensorProblem(
        shape=(1, 1, 2, 1, 1),
        anchors=(4.0, 6.0),
        closure_totals=(10.0,),
        migration_bounds=(1.0, 1.0),
        weights=PopulationObjectiveWeights(death=0.0),
    )
    result, plan = solve_sparse_population(problem, max_iterations=200)
    assert result.population == pytest.approx((4.0, 6.0), abs=1e-5)
    assert sum(result.population) == pytest.approx(10.0)
    assert plan.state_space.anchor_indices == (0, 1)


def test_sparse_state_space_records_transition_and_constraint_groups():
    problem = PopulationTensorProblem(
        shape=(2, 2, 3, 1, 2),
        anchors=(None,) * 24,
        closure_totals=(10.0,) * 4,
        weights=PopulationObjectiveWeights(death=0.0),
    )
    state = build_population_state_space(problem)
    assert len(state.aging_edges) == 8
    assert len(state.birth_edges) == 4
    assert len(state.race_groups) == 12
    assert len(state.closure_groups) == 4


def test_reduced_state_space_smoother_fills_missing_cohort_path():
    problem = PopulationTensorProblem(
        shape=(1, 3, 3, 1, 1),
        anchors=(100.0, None, None, None, 91.0, None, None, None, 84.0),
        hard_anchor_mask=(True, False, False, False, True, False, False, False, True),
        closure_totals=(100.0, 91.0, 84.0),
        migration_bounds=(0.0,) * 9,
        weights=PopulationObjectiveWeights(death=0.0),
    )
    smoothed = solve_population_state_space_smoother(problem)
    assert smoothed.result.telemetry.converged is True
    assert smoothed.result.population[0] == pytest.approx(100.0)
    assert sum(smoothed.result.population[3:6]) == pytest.approx(91.0)
    assert sum(smoothed.result.population[6:9]) == pytest.approx(84.0)
    assert smoothed.as_manifest()["solver_backend"] == "state_space_smoother_reduced_rts"


def test_sparse_population_memory_preflight_aborts():
    problem = PopulationTensorProblem(
        shape=(2, 2, 3, 1, 2), anchors=(None,) * 24,
        weights=PopulationObjectiveWeights(death=0.0),
    )
    with pytest.raises(MemoryPreflightError):
        plan_sparse_population_solver(problem, available_bytes=64)
