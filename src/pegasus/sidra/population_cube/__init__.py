"""The demographic population tensor as a synthetic SIDRA cube (MSD §2.8).

Admits SIDRA table 9606 (population by município × período × sexo/cor/idade) as the
population-denominator tensor, optionally anchored/adjusted with DATASUS vital-flow
contributions (SIM deaths, SINASC births). This package owns the SIDRA-specific data
acquisition and orchestration; the underlying constrained-optimization math (loss
terms, solvers) is generic reconstruction machinery shared with other tensor
instances (age-bin disaggregation, ST-DFM) and lives in ``pegasus.she.reconstruction``.

- ``anchor``: loads SIDRA 9606 population facts (single-anchor and multi-locality forms).
- ``fields``: admits disaggregated SIDRA facts as a canonical Population FieldNode.
- ``build``: orchestrates strata/anchor facts (+ optional SIM death priors) into a
  ``PopulationTensorProblem`` and runs the reconstruction solver.
"""

from __future__ import annotations

from pegasus.sidra.population_cube.anchor import (
    SidraPopulationAnchor,
    load_sidra_population_total_anchor,
    load_sidra_population_totals_frame,
)
from pegasus.sidra.population_cube.build import (
    PopulationTensorBuild,
    solve_population_tensor_from_sidra_anchor,
    solve_population_tensor_from_sidra_strata,
)
from pegasus.sidra.population_cube.fields import build_sidra_demographic_population_fields

__all__ = [
    "SidraPopulationAnchor",
    "load_sidra_population_total_anchor",
    "load_sidra_population_totals_frame",
    "PopulationTensorBuild",
    "solve_population_tensor_from_sidra_strata",
    "solve_population_tensor_from_sidra_anchor",
    "build_sidra_demographic_population_fields",
]
