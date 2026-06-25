from __future__ import annotations

from pathlib import Path

from pegasus.she.population.solvers import dense_national_abort_check, solve_population_tensor_from_sidra_anchor


def run_population_tensor_plan(
    *,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, object]:
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
