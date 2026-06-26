from __future__ import annotations

from pathlib import Path

from pegasus.she.population.solvers import dense_national_abort_check, solve_population_tensor_from_sidra_anchor
from pegasus.she.population.orchestrator import solve_population_tensor_from_sidra_strata


def run_population_tensor_plan(
    *,
    sidra_facts_path: str | Path,
    population_strata_path: str | Path | None = None,
    output_path: str | Path | None = None,
    sim_events_path: str | Path | None = None,
    mode: str = "independent_denominator",
) -> dict[str, object]:
    if population_strata_path is not None:
        if output_path is None:
            raise ValueError("population_strata_path requires output_path for materialized tensor output")
        result = solve_population_tensor_from_sidra_strata(
            population_strata_path=population_strata_path,
            total_anchor_path=sidra_facts_path,
            output_path=output_path,
            mode=mode,
            sim_events_path=sim_events_path,
        )
        return result.as_manifest()

    result = solve_population_tensor_from_sidra_anchor(
        sidra_facts_path=sidra_facts_path,
        mode=mode,
    )
    return result.as_manifest()


def run_population_dense_abort_check(
    *,
    localities: int,
    periods: int,
    strata: int = 1,
) -> dict[str, object]:
    cells = dense_national_abort_check(
        localities=localities,
        periods=periods,
        strata=strata,
    )
    return {"status": "allowed", "cells": cells}


__all__ = ["run_population_tensor_plan", "run_population_dense_abort_check"]
