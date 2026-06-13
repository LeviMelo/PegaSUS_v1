from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.build_efg import (
    build_autonomous_efg_from_manifest,
    build_sim_fixture_efg_run,
)


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


def run_build_autonomous_efg(
    *,
    substrate_manifest: str | Path,
    output_path: str | Path,
    registry_root: str | Path = "config/registries",
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> dict[str, Any]:
    result = build_autonomous_efg_from_manifest(
        substrate_manifest=substrate_manifest,
        output_path=output_path,
        registry_root=registry_root,
        operator_budget=operator_budget,
        operator_mode=operator_mode,
    )
    return {
        "output_path": Path(output_path),
        "efg_id": result.efg_id,
        "field_count": result.field_count,
        "edge_count": result.edge_count,
        "failed_branch_count": len(result.failed_branches),
        "legality_summary": result.legality_summary,
    }
