from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.build_efg import build_sim_fixture_efg_run


def run_build_sim_fixture(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
) -> dict[str, Any]:
    output = build_sim_fixture_efg_run(
        sim_events_path=sim_events_path,
        run_dir=run_dir,
        municipality_cod6=municipality_cod6,
    )
    result = validate_output_bundle(run_dir=str(output))
    return {"run_dir": output, "validation": result}


def run_attach_sidra_denominator(
    *,
    run_dir: str | Path,
    sidra_facts_path: str | Path,
) -> dict[str, Any]:
    output = attach_sidra_population_anchor_to_run(
        run_dir=run_dir,
        sidra_facts_path=sidra_facts_path,
    )
    result = validate_output_bundle(run_dir=str(output))
    return {"run_dir": output, "validation": result}
