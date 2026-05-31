from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from pegasus.datasus.io import write_json
from pegasus.datasus.workflow import process_datasus_local_files
from pegasus.output.bundle import OutputBundle
from pegasus.workflows.compile import compile_run
from pegasus.workflows.mortality import compile_crude_mortality_to_bundle
from pegasus.workflows.sidra_population import prepare_and_import_sidra_population_to_bundle
from pegasus.workflows.sim_compile import compile_sim_death_counts_to_bundle


class LocalSimSidraMortalityResult(BaseModel):
    status: Literal["success"]

    run_dir: str
    intent_path: str

    sim_raw_path: str
    sim_processed_path: str | None
    normalization_input_kind: Literal["raw", "processed"]

    sidra_spec_path: str
    sidra_out_root: str
    sidra_denominator_path: str
    sidra_projection_report_path: str

    normalized_sim_path: str
    sim_workflow_manifest_path: str

    death_count_field_id: str
    population_field_id: str
    crude_mortality_field_id: str

    geography: Literal["residence", "occurrence"]
    source_label: str
    summary_path: str


def run_local_sim_sidra_mortality(
    *,
    root: str | Path,
    intent_path: str | Path,
    raw_sim_path: str | Path,
    sidra_population_spec_path: str | Path,
    sidra_out_root: str | Path,
    sidra_denominator_output_path: str | Path,
    processed_sim_path: str | Path | None = None,
    normalization_input_kind: Literal["raw", "processed"] = "raw",
    geography: Literal["residence", "occurrence"] = "residence",
    sidra_table_id: str | None = None,
    sidra_variable_id: str | None = None,
    population_source_label: str = "SIDRA_population",
    required_locality_level_id: str | None = "6",
    use_sidra_cache: bool = True,
) -> LocalSimSidraMortalityResult:
    root = Path(root).resolve()
    intent_path = Path(intent_path).resolve()
    raw_sim_path = Path(raw_sim_path).resolve()
    processed_sim_resolved = (
        None if processed_sim_path is None else Path(processed_sim_path).resolve()
    )

    sidra_population_spec_path = Path(sidra_population_spec_path).resolve()
    sidra_out_root = Path(sidra_out_root).resolve()
    sidra_denominator_output_path = Path(sidra_denominator_output_path).resolve()

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

    sidra_import = prepare_and_import_sidra_population_to_bundle(
        root=root,
        run_dir=run_dir,
        spec_path=sidra_population_spec_path,
        sidra_out_root=sidra_out_root,
        denominator_output_path=sidra_denominator_output_path,
        table_id=sidra_table_id,
        variable_id=sidra_variable_id,
        source_label=population_source_label,
        required_locality_level_id=required_locality_level_id,
        use_cache=use_sidra_cache,
    )

    crude_mortality_field_id = compile_crude_mortality_to_bundle(
        run_dir=run_dir,
        registry_dir=registry_dir,
        death_count_field_id=death_count_field_id,
        population_field_id=sidra_import.population_field_id,
        scale=100_000.0,
    )

    bundle.validate_minimal()

    summary_path = run_dir / "Tables" / "workflows" / "local_sim_sidra_mortality.json"

    result = LocalSimSidraMortalityResult(
        status="success",
        run_dir=str(run_dir),
        intent_path=str(intent_path),
        sim_raw_path=str(raw_sim_path),
        sim_processed_path=None
        if processed_sim_resolved is None
        else str(processed_sim_resolved),
        normalization_input_kind=normalization_input_kind,
        sidra_spec_path=str(sidra_population_spec_path),
        sidra_out_root=str(sidra_out_root),
        sidra_denominator_path=sidra_import.denominator_path,
        sidra_projection_report_path=sidra_import.projection_report_path,
        normalized_sim_path=sim_manifest.normalized_path,
        sim_workflow_manifest_path=str(sim_workflow_dir / "manifest.json"),
        death_count_field_id=death_count_field_id,
        population_field_id=sidra_import.population_field_id,
        crude_mortality_field_id=crude_mortality_field_id,
        geography=geography,
        source_label=population_source_label,
        summary_path=str(summary_path),
    )

    write_json(summary_path, result)

    return result