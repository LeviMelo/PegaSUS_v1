from __future__ import annotations

from pathlib import Path

from pegasus.output.validate import validate_output_bundle
from pegasus.she.population.solvers import dense_national_abort_check, solve_population_tensor_from_sidra_anchor


def run_population_tensor_fixture(
    *,
    sidra_facts_path: str | Path,
    run_dir: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, object]:
    out = write_population_tensor_fixture_bundle(sidra_facts_path=sidra_facts_path, run_dir=run_dir, mode=mode)
    validation = validate_output_bundle(run_dir=str(out))
    return {"run_dir": out, "validation": validation}


def run_population_tensor_plan(
    *,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, object]:
    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=sidra_facts_path, mode=mode)
    return result.as_manifest()


def run_population_dense_abort_check(*, localities: int, periods: int, strata: int = 1) -> dict[str, object]:
    cells = dense_national_abort_check(localities=localities, periods=periods, strata=strata)
    return {"status": "allowed", "cells": cells}


def run_attach_population_tensor_compile_fields(*args, **kwargs):
    raise RuntimeError(
        "Population tensor manual workflow is retired. "
        "Population denominators must enter through SHE/EFG/PIRS contracts."
    )

