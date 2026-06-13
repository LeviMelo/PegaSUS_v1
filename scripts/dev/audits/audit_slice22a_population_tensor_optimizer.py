from __future__ import annotations

import json

from pegasus.she.population.schema import PopulationObjectiveWeights, PopulationTensorProblem
from pegasus.she.population.solvers import solve_population_tensor_problem


def main() -> int:
    problem = PopulationTensorProblem(
        shape=(1, 2, 3, 1, 1),
        anchors=(60.0, 30.0, 10.0, None, None, None),
        hard_anchor_mask=(True, True, True, False, False, False),
        closure_totals=(100.0, 150.0),
        births=(50.0, None),
        death_rates=(0.0,) * 6,
        migration_bounds=(0.0,) * 6,
        weights=PopulationObjectiveWeights(aging=2.0, birth=2.0, age_smooth=0.0),
    )
    result = solve_population_tensor_problem(problem)
    checks = {
        "converged": result.telemetry.converged,
        "objective_reduced": result.telemetry.final_objective < result.telemetry.initial_objective,
        "closure_preserved": abs(sum(result.population[3:]) - 150.0) < 1e-6,
        "cohort_reconstructed": all(abs(a - b) < 0.1 for a, b in zip(result.population[3:], (50.0, 60.0, 40.0), strict=True)),
    }
    print(json.dumps({"slice": "22A", "checks": checks, "telemetry": result.telemetry.as_manifest()}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
