"""Population tensor solver orchestration (MSD §2.8).

This package turns real disaggregated SIDRA 9606 facts into a
``PopulationTensorProblem`` over ``(municipality, year, age_group, sex, race)`` and
runs the existing analytic solver. It is intentionally projection-aware: only source
categories that can be mapped through ``demographic/demographic_axis_maps.yaml`` become strata.
Unmapped or unknown categories are excluded rather than silently relabelled.

Decomposed (behavior-preserving) from the former single ``build.py`` module into a
layered package -- ``indexing`` (flat-cell math) <- ``layer1`` (closed-form prior mean)
<- ``priors`` (flow priors) <- ``closure`` (closure panel); ``strata``,
``projection_envelope``, ``flows`` are independent; ``orchestrator`` sits on top. Every
symbol the old module exposed (including the underscore helpers imported by tests) is
re-exported here, so ``from pegasus.denominators.population.build import <X>`` is unchanged.
"""

from __future__ import annotations

from pegasus.denominators.population.build.indexing import *  # noqa: F401,F403
from pegasus.denominators.population.build.layer1 import *  # noqa: F401,F403
from pegasus.denominators.population.build.strata import *  # noqa: F401,F403
from pegasus.denominators.population.build.priors import *  # noqa: F401,F403
from pegasus.denominators.population.build.closure import *  # noqa: F401,F403
from pegasus.denominators.population.build.flows import *  # noqa: F401,F403
from pegasus.denominators.population.build.projection_envelope import *  # noqa: F401,F403
from pegasus.denominators.population.build.orchestrator import *  # noqa: F401,F403

from pegasus.denominators.population.build.indexing import (
    _birth_cell_index,
    _cell_index,
    _census_count_arrays,
)
from pegasus.denominators.population.build.layer1 import (
    _interpolate_shares,
    interpolate_census_composition,
)
from pegasus.denominators.population.build.strata import (
    AXES,
    AXIS_CLASSIFICATIONS,
    _canonical_stratum,
    _category_by_classification,
    _census_2000_records_from_facts,
    _pairs,
    _read_population_strata,
)
from pegasus.denominators.population.build.priors import (
    _bridge_race_stratified_counts,
    _census_race_composition_prior,
    _coalesce_race_columns,
    _resolve_geo_year_columns,
    _sim_death_priors,
    _sinasc_birth_priors,
    _stratify_age_column,
    _stratify_sex_column,
)
from pegasus.denominators.population.build.closure import (
    _datasus_event_totals,
    _migration_residual_totals,
    _reanchor_closure_single_vintage,
    _sidra_vital_totals,
)
from pegasus.denominators.population.build.flows import _reconstruct_and_persist_migration_flows
from pegasus.denominators.population.build.projection_envelope import (
    PROJECTION_H_SOFT,
    PROJECTION_UNCERTAINTY_PER_YEAR,
    _classify_projection_years,
)
from pegasus.denominators.population.build.orchestrator import (
    SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE,
    SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE,
    SIDRA_CIVIL_REGISTRY_DEATHS_TABLE,
    SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE,
    PopulationTensorBuild,
    solve_population_tensor_from_sidra_anchor,
    solve_population_tensor_from_sidra_strata,
)

__all__ = [
    # orchestrator public surface
    "PopulationTensorBuild",
    "solve_population_tensor_from_sidra_strata",
    "solve_population_tensor_from_sidra_anchor",
    "SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE",
    "SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE",
    "SIDRA_CIVIL_REGISTRY_DEATHS_TABLE",
    "SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE",
    # layer1
    "interpolate_census_composition",
    "_interpolate_shares",
    # indexing
    "_cell_index",
    "_birth_cell_index",
    "_census_count_arrays",
    # strata
    "AXES",
    "AXIS_CLASSIFICATIONS",
    "_pairs",
    "_category_by_classification",
    "_canonical_stratum",
    "_read_population_strata",
    "_census_2000_records_from_facts",
    # priors
    "_resolve_geo_year_columns",
    "_stratify_sex_column",
    "_stratify_age_column",
    "_coalesce_race_columns",
    "_bridge_race_stratified_counts",
    "_sim_death_priors",
    "_sinasc_birth_priors",
    "_census_race_composition_prior",
    # closure
    "_sidra_vital_totals",
    "_datasus_event_totals",
    "_migration_residual_totals",
    "_reanchor_closure_single_vintage",
    # flows
    "_reconstruct_and_persist_migration_flows",
    # projection_envelope
    "PROJECTION_H_SOFT",
    "PROJECTION_UNCERTAINTY_PER_YEAR",
    "_classify_projection_years",
]
