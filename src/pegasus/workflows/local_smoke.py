from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from pegasus.datasus.io import write_json
from pegasus.datasus.workflow import process_datasus_local_files
from pegasus.output.bundle import OutputBundle
from pegasus.workflows.compile import compile_run
from pegasus.workflows.mortality import compile_crude_mortality_to_bundle
from pegasus.workflows.population import import_population_to_bundle
from pegasus.workflows.sim_compile import compile_sim_death_counts_to_bundle


class LocalSimMortalitySmokeResult(BaseModel):
    status: Literal["success"]
    run_dir: str
    intent_path: str

    sim_raw_path: str
    sim_processed_path: str | None
    population_path: str

    sim_workflow_dir: str
    sim_workflow_manifest_path: str
    normalized_sim_path: str

    death_count_field_id: str
    population_field_id: str
    crude_mortality_field_id: str

    geography: Literal["residence", "occurrence"]
    normalization_input_kind: Literal["raw", "processed"]
    source_label: str

    summary_path: str


def run_local_sim_mortality_smoke(
    *,
    root: str | Path,
    intent_path: str | Path,
    raw_sim_path: str | Path,
    population_path: str | Path,
    processed_sim_path: str | Path | None = None,
    normalization_input_kind: Literal["raw", "processed"] = "raw",
    geography: Literal["residence", "occurrence"] = "residence",
    population_source_label: str = "imported_population_fixture",
) -> LocalSimMortalitySmokeResult:
    """Run the first local PegaSUS mortality vertical slice.

    This composes already-implemented production-path modules. It deliberately
    does not introduce new analytical semantics.
    """
    root = Path(root).resolve()
    intent_path = Path(intent_path).resolve()
    raw_sim_path = Path(raw_sim_path).resolve()
    population_path = Path(population_path).resolve()
    processed_sim_resolved = (
        None if processed_sim_path is None else Path(processed_sim_path).resolve()
    )

    registry_dir = root / "config" / "registries"

    run_dir = compile_run(
        intent_path=intent_path,
        root=root,
    )

    bundle = OutputBundle(run_dir)
    bundle.validate_minimal()

    sim_workflow_dir = run_dir / "Tables" / "datasus" / "SIM-DO" / "local_workflow"

    sim_manifest = process_datasus_local_files(
        source_system="SIM-DO",
        raw_path=raw_sim_path,
        processed_path=processed_sim_resolved,
        out_root=sim_workflow_dir,
        normalization_input_kind=normalization_input_kind,
    )

    death_count_field_id = compile_sim_death_counts_to_bundle(
        normalized_path=sim_manifest.normalized_path,
        run_dir=run_dir,
        registry_dir=registry_dir,
        geography=geography,
    )

    population_field_id = import_population_to_bundle(
        population_path=population_path,
        run_dir=run_dir,
        registry_dir=registry_dir,
        source_label=population_source_label,
    )

    crude_mortality_field_id = compile_crude_mortality_to_bundle(
        run_dir=run_dir,
        registry_dir=registry_dir,
        death_count_field_id=death_count_field_id,
        population_field_id=population_field_id,
        scale=100_000.0,
    )

    bundle.validate_minimal()

    summary_path = run_dir / "Tables" / "workflows" / "local_sim_mortality_smoke.json"

    result = LocalSimMortalitySmokeResult(
        status="success",
        run_dir=str(run_dir),
        intent_path=str(intent_path),
        sim_raw_path=str(raw_sim_path),
        sim_processed_path=None
        if processed_sim_resolved is None
        else str(processed_sim_resolved),
        population_path=str(population_path),
        sim_workflow_dir=str(sim_workflow_dir),
        sim_workflow_manifest_path=str(sim_workflow_dir / "manifest.json"),
        normalized_sim_path=sim_manifest.normalized_path,
        death_count_field_id=death_count_field_id,
        population_field_id=population_field_id,
        crude_mortality_field_id=crude_mortality_field_id,
        geography=geography,
        normalization_input_kind=normalization_input_kind,
        source_label=population_source_label,
        summary_path=str(summary_path),
    )

    write_json(summary_path, result)

    return result