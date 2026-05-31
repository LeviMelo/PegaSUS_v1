from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from pegasus.datasus.io import write_json
from pegasus.sidra.population import SIDRAPopulationProjectionReport
from pegasus.sidra.population import write_sidra_population_denominator_table
from pegasus.sidra.workflow import fetch_sidra_query_to_disk
from pegasus.workflows.population import import_population_to_bundle


class SIDRAPopulationPrepareResult(BaseModel):
    facts_path: str
    denominator_path: str
    projection_report_path: str
    n_population_rows: int
    warnings: list[str]


class SIDRAPopulationBundleImportResult(BaseModel):
    facts_path: str
    denominator_path: str
    projection_report_path: str
    population_field_id: str
    n_population_rows: int
    warnings: list[str]


def prepare_sidra_population_denominator(
    *,
    root: str | Path,
    spec_path: str | Path,
    sidra_out_root: str | Path,
    denominator_output_path: str | Path,
    table_id: str | None = None,
    variable_id: str | None = None,
    source_label: str = "SIDRA_population",
    required_locality_level_id: str | None = "6",
    allow_empty: bool = False,
    use_cache: bool = True,
) -> SIDRAPopulationPrepareResult:
    fetch_manifest = fetch_sidra_query_to_disk(
        root=root,
        spec_path=spec_path,
        out_root=sidra_out_root,
        allow_empty=allow_empty,
        use_cache=use_cache,
    )

    projection_report_path = Path(denominator_output_path).with_suffix(
        ".projection_report.json"
    )

    denominator_path, report = write_sidra_population_denominator_table(
        facts_path=fetch_manifest.facts_path,
        output_path=denominator_output_path,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
        required_locality_level_id=required_locality_level_id,
        report_path=projection_report_path,
    )

    return SIDRAPopulationPrepareResult(
        facts_path=fetch_manifest.facts_path,
        denominator_path=str(denominator_path),
        projection_report_path=str(projection_report_path),
        n_population_rows=report.n_output_rows,
        warnings=[*fetch_manifest.warnings, *report.warnings],
    )


def prepare_and_import_sidra_population_to_bundle(
    *,
    root: str | Path,
    run_dir: str | Path,
    spec_path: str | Path,
    sidra_out_root: str | Path,
    denominator_output_path: str | Path,
    table_id: str | None = None,
    variable_id: str | None = None,
    source_label: str = "SIDRA_population",
    required_locality_level_id: str | None = "6",
    allow_empty: bool = False,
    use_cache: bool = True,
) -> SIDRAPopulationBundleImportResult:
    prepared = prepare_sidra_population_denominator(
        root=root,
        spec_path=spec_path,
        sidra_out_root=sidra_out_root,
        denominator_output_path=denominator_output_path,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
        required_locality_level_id=required_locality_level_id,
        allow_empty=allow_empty,
        use_cache=use_cache,
    )

    population_field_id = import_population_to_bundle(
        population_path=prepared.denominator_path,
        run_dir=run_dir,
        registry_dir=Path(root) / "config" / "registries",
        source_label=source_label,
    )

    result = SIDRAPopulationBundleImportResult(
        facts_path=prepared.facts_path,
        denominator_path=prepared.denominator_path,
        projection_report_path=prepared.projection_report_path,
        population_field_id=population_field_id,
        n_population_rows=prepared.n_population_rows,
        warnings=prepared.warnings,
    )

    summary_path = Path(run_dir) / "Tables" / "workflows" / "sidra_population_import.json"
    write_json(summary_path, result)

    return result