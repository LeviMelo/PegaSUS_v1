from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from pegasus import __version__
from pegasus.compute.torch_backend import inspect_torch
from pegasus.core.config import load_config, resolve_project_paths
from pegasus.core.logging import configure_logging
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set
from pegasus.workflows.compile import compile_run
from pegasus.datasus.normalize import normalize_datasus_table
from pegasus.datasus.workflow import process_sim_do_local_files
from pegasus.datasus.adapters.registry import registered_datasus_systems
from pegasus.workflows.sim_compile import compile_sim_death_counts_to_bundle

from pegasus.output.field_store import list_fields
from pegasus.workflows.mortality import compile_crude_mortality_to_bundle
from pegasus.workflows.population import import_population_to_bundle

from pegasus.workflows.local_smoke import run_local_sim_mortality_smoke

from pegasus.sidra.workflow import fetch_sidra_query_to_disk

from pegasus.sidra.population import write_sidra_population_denominator_table

from pegasus.datasus.io import write_json
from pegasus.sidra.diagnostics import diagnose_sidra_facts
from pegasus.workflows.sidra_population import (
    prepare_and_import_sidra_population_to_bundle,
    prepare_sidra_population_denominator,
)

from pegasus.workflows.local_sidra_mortality import run_local_sim_sidra_mortality

from pegasus.datasus.client_microdatasus import MicrodatasusRequest, ingest_microdatasus

app = typer.Typer(help="PegaSUS epidemiological compiler CLI.")
console = Console()


@app.callback()
def main() -> None:
    configure_logging()


@app.command()
def doctor(root: Path = typer.Option(Path("."), help="Project root.")) -> None:
    """Check local development environment."""
    root = root.resolve()

    table = Table(title="PegaSUS doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")

    table.add_row("PegaSUS", "OK", __version__)
    table.add_row("Python", "OK", platform.python_version())
    table.add_row("Platform", "OK", platform.platform())

    try:
        config = load_config(root)
        table.add_row("Config", "OK", f"Project: {config.project.name}")
    except Exception as exc:
        table.add_row("Config", "FAIL", str(exc))

    try:
        paths = resolve_project_paths(root)
        paths.ensure_all()
        table.add_row("Paths", "OK", str(paths.root))
    except Exception as exc:
        table.add_row("Paths", "FAIL", str(exc))

    try:
        registry_set = load_registry_set(root / "config" / "registries")
        validate_registry_set(registry_set)
        table.add_row(
            "Registries",
            "OK",
            f"{len(registry_set.registries)} registries; "
            f"manifest={registry_set.manifest.registry_set.version}",
        )
    except Exception as exc:
        table.add_row("Registries", "FAIL", str(exc))

    torch_status = inspect_torch()
    if torch_status.installed:
        cuda_status = "CUDA OK" if torch_status.cuda_available else "CPU only"
        details = (
            f"torch={torch_status.torch_version}; "
            f"cuda={torch_status.cuda_version}; "
            f"device={torch_status.device_name}"
        )
        table.add_row("PyTorch", cuda_status, details)
    else:
        table.add_row("PyTorch", "FAIL", torch_status.error or "not installed")

    rscript = shutil.which("Rscript")
    if rscript:
        table.add_row("Rscript", "OK", rscript)
    else:
        table.add_row("Rscript", "WARN", "Rscript not found on PATH")

    console.print(table)


@app.command("validate-config")
def validate_config(root: Path = typer.Option(Path("."), help="Project root.")) -> None:
    """Validate project configuration files."""
    config = load_config(root)
    console.print_json(json.dumps(config.model_dump(mode="json"), ensure_ascii=False))


@app.command("validate-registries")
def validate_registries(root: Path = typer.Option(Path("."), help="Project root.")) -> None:
    """Validate registry manifest and registry files."""
    registry_set = load_registry_set(root / "config" / "registries")
    warnings = validate_registry_set(registry_set)

    table = Table(title="Registry validation")
    table.add_column("Registry")
    table.add_column("SHA256")
    table.add_column("Path")

    table.add_row("registry_manifest", registry_set.manifest_sha256, str(registry_set.manifest_path))

    for name, loaded in registry_set.registries.items():
        table.add_row(name, loaded.sha256, str(loaded.path))

    console.print(table)

    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"- {warning}")

    console.print("[green]Registry validation passed.[/green]")


@app.command("compile")
def compile_command(
    intent: Path = typer.Option(..., "--intent", "-i", help="Path to intent JSON."),
    root: Path = typer.Option(Path("."), help="Project root."),
) -> None:
    """Create a PegaSUS run bundle for the provided intent."""
    run_dir = compile_run(intent_path=intent, root=root)
    console.print(f"[green]Run bundle created:[/green] {run_dir}")
    
    
@app.command("datasus-normalize")
def datasus_normalize(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input CSV/Parquet table."),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output normalized Parquet path."),
    source_system: str = typer.Option(..., "--system", help="DATASUS source system, e.g. SIM-DO."),
    source_manifest_hash: str = typer.Option("", help="Source acquisition manifest hash."),
) -> None:
    """Normalize a DATASUS table into a PegaSUS normalized event table."""
    out = normalize_datasus_table(
        input_path=input_path,
        output_path=output_path,
        source_system=source_system,  # type: ignore[arg-type]
        source_manifest_hash=source_manifest_hash,
    )
    console.print(f"[green]Normalized table written:[/green] {out}")
    
    
@app.command("datasus-process-local")
def datasus_process_local(
    source_system: str = typer.Option(..., "--system", help="DATASUS source system."),
    raw_path: Path = typer.Option(..., "--raw", help="Raw DATASUS CSV/Parquet path."),
    processed_path: Path | None = typer.Option(
        None,
        "--processed",
        help="Optional processed DATASUS CSV/Parquet path.",
    ),
    out_root: Path = typer.Option(
        ...,
        "--out-root",
        help="Output directory for profiles, comparison, manifest, and normalized Parquet.",
    ),
    normalization_input_kind: str = typer.Option(
        "raw",
        "--normalization-input",
        help="Which artifact to normalize: raw or processed.",
    ),
) -> None:
    """Profile, compare, and normalize local DATASUS files through a source adapter."""
    from pegasus.datasus.workflow import process_datasus_local_files

    if normalization_input_kind not in {"raw", "processed"}:
        raise typer.BadParameter("normalization-input must be either 'raw' or 'processed'.")

    manifest = process_datasus_local_files(
        source_system=source_system,
        raw_path=raw_path,
        processed_path=processed_path,
        out_root=out_root,
        normalization_input_kind=normalization_input_kind,  # type: ignore[arg-type]
    )

    console.print("[green]DATASUS local workflow complete.[/green]")
    console.print(f"source_system: {manifest.source_system}")
    console.print(f"workflow_id: {manifest.workflow_id}")
    console.print(f"normalized: {manifest.normalized_path}")


@app.command("datasus-process-local-sim")
def datasus_process_local_sim(
    raw_path: Path = typer.Option(..., "--raw", help="Raw SIM-DO CSV/Parquet path."),
    processed_path: Path | None = typer.Option(
        None,
        "--processed",
        help="Optional processed SIM-DO CSV/Parquet path.",
    ),
    out_root: Path = typer.Option(
        ...,
        "--out-root",
        help="Output directory for profiles, comparison, manifest, and normalized Parquet.",
    ),
    normalization_input_kind: str = typer.Option(
        "raw",
        "--normalization-input",
        help="Which artifact to normalize: raw or processed.",
    ),
) -> None:
    """Alias for datasus-process-local --system SIM-DO."""
    from pegasus.datasus.workflow import process_sim_do_local_files

    if normalization_input_kind not in {"raw", "processed"}:
        raise typer.BadParameter("normalization-input must be either 'raw' or 'processed'.")

    manifest = process_sim_do_local_files(
        raw_path=raw_path,
        processed_path=processed_path,
        out_root=out_root,
        normalization_input_kind=normalization_input_kind,  # type: ignore[arg-type]
    )

    console.print("[green]SIM-DO local workflow complete.[/green]")
    console.print(f"workflow_id: {manifest.workflow_id}")
    console.print(f"normalized: {manifest.normalized_path}")
    
    
@app.command("datasus-adapters")
def datasus_adapters() -> None:
    """List registered DATASUS source adapters."""
    table = Table(title="Registered DATASUS adapters")
    table.add_column("Source system")

    for system in registered_datasus_systems():
        table.add_row(system)

    console.print(table)
    
    
@app.command("sim-compile-death-counts")
def sim_compile_death_counts(
    normalized_path: Path = typer.Option(
        ...,
        "--normalized",
        help="Normalized SIM-DO Parquet path.",
    ),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Existing PegaSUS run bundle directory.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    geography: str = typer.Option(
        "residence",
        "--geography",
        help="residence or occurrence.",
    ),
) -> None:
    """Compile normalized SIM-DO events into a materialized death-count field."""
    if geography not in {"residence", "occurrence"}:
        raise typer.BadParameter("geography must be residence or occurrence.")

    field_id = compile_sim_death_counts_to_bundle(
        normalized_path=normalized_path,
        run_dir=run_dir,
        registry_dir=root / "config" / "registries",
        geography=geography,  # type: ignore[arg-type]
    )

    console.print("[green]SIM-DO death-count field compiled.[/green]")
    console.print(f"field_id: {field_id}")
    
    
@app.command("fields")
def fields(
    run_dir: Path = typer.Option(..., "--run-dir", help="PegaSUS run bundle directory."),
) -> None:
    """List materialized fields in a run bundle."""
    df = list_fields(run_dir)

    if df.is_empty():
        console.print("[yellow]No fields materialized yet.[/yellow]")
        return

    console.print(df)


@app.command("population-import")
def population_import(
    population_path: Path = typer.Option(
        ...,
        "--population",
        help="CSV/Parquet with year, municipality_cod6, population.",
    ),
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Existing PegaSUS run bundle directory.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    source_label: str = typer.Option(
        "imported_population_fixture",
        "--source-label",
        help="Source label for imported denominator.",
    ),
) -> None:
    """Import a population denominator table into a run bundle."""
    field_id = import_population_to_bundle(
        population_path=population_path,
        run_dir=run_dir,
        registry_dir=root / "config" / "registries",
        source_label=source_label,
    )

    console.print("[green]Population denominator field imported.[/green]")
    console.print(f"field_id: {field_id}")


@app.command("mortality-compile-crude")
def mortality_compile_crude(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        help="Existing PegaSUS run bundle directory.",
    ),
    death_count_field_id: str = typer.Option(
        ...,
        "--death-field",
        help="Deaths/counts numerator field ID.",
    ),
    population_field_id: str = typer.Option(
        ...,
        "--population-field",
        help="Population/person-years denominator field ID.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    scale: float = typer.Option(100_000.0, "--scale", help="Rate scale."),
) -> None:
    """Compile crude mortality rate from death count and population fields."""
    field_id = compile_crude_mortality_to_bundle(
        run_dir=run_dir,
        registry_dir=root / "config" / "registries",
        death_count_field_id=death_count_field_id,
        population_field_id=population_field_id,
        scale=scale,
    )

    console.print("[green]Crude mortality-rate field compiled.[/green]")
    console.print(f"field_id: {field_id}")
    
    
@app.command("local-sim-mortality-smoke")
def local_sim_mortality_smoke(
    raw_sim: Path = typer.Option(
        ...,
        "--raw-sim",
        help="Raw/local SIM-DO CSV or Parquet file.",
    ),
    population: Path = typer.Option(
        ...,
        "--population",
        help="Population CSV/Parquet with year, municipality_cod6, population.",
    ),
    intent: Path = typer.Option(
        Path("config/intents/alagoas_smoke.json"),
        "--intent",
        help="Intent JSON file.",
    ),
    processed_sim: Path | None = typer.Option(
        None,
        "--processed-sim",
        help="Optional processed SIM-DO CSV or Parquet file.",
    ),
    normalization_input_kind: str = typer.Option(
        "raw",
        "--normalization-input",
        help="raw or processed.",
    ),
    geography: str = typer.Option(
        "residence",
        "--geography",
        help="residence or occurrence.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    population_source_label: str = typer.Option(
        "imported_population_fixture",
        "--population-source-label",
        help="Label for imported population denominator.",
    ),
) -> None:
    """Run local SIM-DO → death count → population → crude mortality vertical slice."""
    if normalization_input_kind not in {"raw", "processed"}:
        raise typer.BadParameter("normalization-input must be raw or processed.")

    if geography not in {"residence", "occurrence"}:
        raise typer.BadParameter("geography must be residence or occurrence.")

    result = run_local_sim_mortality_smoke(
        root=root,
        intent_path=intent,
        raw_sim_path=raw_sim,
        processed_sim_path=processed_sim,
        population_path=population,
        normalization_input_kind=normalization_input_kind,  # type: ignore[arg-type]
        geography=geography,  # type: ignore[arg-type]
        population_source_label=population_source_label,
    )

    console.print("[green]Local SIM mortality smoke run complete.[/green]")
    console.print(f"run_dir: {result.run_dir}")
    console.print(f"normalized_sim: {result.normalized_sim_path}")
    console.print(f"death_count_field_id: {result.death_count_field_id}")
    console.print(f"population_field_id: {result.population_field_id}")
    console.print(f"crude_mortality_field_id: {result.crude_mortality_field_id}")
    console.print(f"summary: {result.summary_path}")
    

@app.command("sidra-fetch")
def sidra_fetch(
    spec: Path = typer.Option(..., "--spec", help="SIDRA query spec YAML/JSON."),
    out_root: Path = typer.Option(..., "--out-root", help="Output directory."),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    allow_empty: bool = typer.Option(
        False,
        "--allow-empty",
        help="Allow zero-fact SIDRA results without failing.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Force HTTP request even when raw JSON cache exists.",
    ),
) -> None:
    """Fetch a SIDRA query, persist raw JSON, normalize facts, and write manifest."""
    manifest = fetch_sidra_query_to_disk(
        root=root,
        spec_path=spec,
        out_root=out_root,
        allow_empty=allow_empty,
        use_cache=not no_cache,
    )

    console.print("[green]SIDRA fetch complete.[/green]")
    console.print(f"table_id: {manifest.table_id}")
    console.print(f"periods: {manifest.periods}")
    console.print(f"variables: {manifest.variables}")
    console.print(f"view: {manifest.view}")
    console.print(f"cache_hit: {manifest.cache_hit}")
    console.print(f"download_bytes: {manifest.download_bytes}")
    console.print(f"n_raw_top_level_items: {manifest.n_raw_top_level_items}")
    console.print(f"n_facts: {manifest.n_facts}")
    console.print(f"url: {manifest.url}")
    console.print(f"raw_json: {manifest.raw_json_path}")
    console.print(f"facts: {manifest.facts_path}")
    console.print(f"manifest: {manifest.manifest_path}")

    if manifest.timings_seconds:
        table = Table(title="SIDRA timings")
        table.add_column("Stage")
        table.add_column("Seconds", justify="right")

        for stage, seconds in manifest.timings_seconds.items():
            table.add_row(stage, f"{seconds:.6f}")

        console.print(table)

    if manifest.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in manifest.warnings:
            console.print(f"- {warning}")
            
            
            
@app.command("sidra-population-export")
def sidra_population_export(
    facts_path: Path = typer.Option(
        ...,
        "--facts",
        help="SIDRA normalized facts Parquet.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        help="Output population denominator Parquet.",
    ),
    table_id: str | None = typer.Option(
        None,
        "--table-id",
        help="Optional SIDRA table filter.",
    ),
    variable_id: str | None = typer.Option(
        None,
        "--variable-id",
        help="Optional SIDRA variable filter.",
    ),
    source_label: str = typer.Option(
        "SIDRA_population",
        "--source-label",
        help="Denominator source label.",
    ),
    required_locality_level_id: str | None = typer.Option(
        "6",
        "--required-locality-level-id",
        help="SIDRA territorial level code required for denominator export. Use 6 for Município.",
    ),
) -> None:
    """Convert SIDRA population facts into a denominator table."""
    out, report = write_sidra_population_denominator_table(
        facts_path=facts_path,
        output_path=output_path,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
        required_locality_level_id=required_locality_level_id,
    )

    console.print("[green]SIDRA population denominator exported.[/green]")
    console.print(f"output: {out}")
    console.print(f"rows: {report.n_output_rows}")
    console.print(f"projection_report: {Path(out).with_suffix('.projection_report.json')}")

    table = Table(title="SIDRA population projection")
    table.add_column("Stage")
    table.add_column("Rows", justify="right")

    table.add_row("input_facts", str(report.n_input_facts))
    table.add_row("after_table_filter", str(report.n_after_table_filter))
    table.add_row("after_variable_filter", str(report.n_after_variable_filter))
    table.add_row("after_locality_level_filter", str(report.n_after_locality_level_filter))
    table.add_row("after_value_state_filter", str(report.n_after_value_state_filter))
    table.add_row("output_rows", str(report.n_output_rows))

    console.print(table)

    if report.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in report.warnings:
            console.print(f"- {warning}")
            


@app.command("sidra-diagnose-facts")
def sidra_diagnose_facts(
    facts_path: Path = typer.Option(
        ...,
        "--facts",
        help="SIDRA normalized facts Parquet.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional diagnostics JSON output path.",
    ),
) -> None:
    """Diagnose normalized SIDRA facts."""
    diagnostics = diagnose_sidra_facts(facts_path)

    table = Table(title="SIDRA facts diagnostics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("rows", str(diagnostics.n_rows))
    table.add_row("unique_tables", str(diagnostics.n_unique_tables))
    table.add_row("unique_variables", str(diagnostics.n_unique_variables))
    table.add_row("unique_periods", str(diagnostics.n_unique_periods))
    table.add_row("unique_localities", str(diagnostics.n_unique_localities))
    table.add_row("valid_values", str(diagnostics.n_valid_value))
    table.add_row("null_values", str(diagnostics.n_null_value))
    table.add_row("duplicate_measure_keys", str(diagnostics.n_duplicate_measure_keys))

    console.print(table)

    console.print("[bold]Locality levels:[/bold]")
    console.print(diagnostics.locality_level_counts)

    console.print("[bold]Value states:[/bold]")
    console.print(diagnostics.value_state_counts)

    if diagnostics.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in diagnostics.warnings:
            console.print(f"- {warning}")

    if output_json is not None:
        write_json(output_json, diagnostics)
        console.print(f"[green]Diagnostics written:[/green] {output_json}")
        
        
        
@app.command("sidra-population-prepare")
def sidra_population_prepare(
    spec: Path = typer.Option(..., "--spec", help="SIDRA population query spec."),
    sidra_out_root: Path = typer.Option(..., "--sidra-out-root", help="SIDRA raw/facts output dir."),
    denominator_output: Path = typer.Option(
        ...,
        "--denominator-output",
        help="Output denominator Parquet.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    table_id: str | None = typer.Option(None, "--table-id"),
    variable_id: str | None = typer.Option(None, "--variable-id"),
    source_label: str = typer.Option("SIDRA_population", "--source-label"),
    required_locality_level_id: str | None = typer.Option(
        "6",
        "--required-locality-level-id",
    ),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Fetch SIDRA and produce a denominator table in one command."""
    result = prepare_sidra_population_denominator(
        root=root,
        spec_path=spec,
        sidra_out_root=sidra_out_root,
        denominator_output_path=denominator_output,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
        required_locality_level_id=required_locality_level_id,
        use_cache=not no_cache,
    )

    console.print("[green]SIDRA population denominator prepared.[/green]")
    console.print(f"facts: {result.facts_path}")
    console.print(f"denominator: {result.denominator_path}")
    console.print(f"projection_report: {result.projection_report_path}")
    console.print(f"rows: {result.n_population_rows}")

    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"- {warning}")
            


@app.command("sidra-population-import-bundle")
def sidra_population_import_bundle(
    run_dir: Path = typer.Option(..., "--run-dir", help="Existing PegaSUS run bundle."),
    spec: Path = typer.Option(..., "--spec", help="SIDRA population query spec."),
    sidra_out_root: Path = typer.Option(..., "--sidra-out-root", help="SIDRA raw/facts output dir."),
    denominator_output: Path = typer.Option(
        ...,
        "--denominator-output",
        help="Output denominator Parquet.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    table_id: str | None = typer.Option(None, "--table-id"),
    variable_id: str | None = typer.Option(None, "--variable-id"),
    source_label: str = typer.Option("SIDRA_population", "--source-label"),
    required_locality_level_id: str | None = typer.Option(
        "6",
        "--required-locality-level-id",
    ),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Fetch SIDRA population, export denominator, and import it into a run bundle."""
    result = prepare_and_import_sidra_population_to_bundle(
        root=root,
        run_dir=run_dir,
        spec_path=spec,
        sidra_out_root=sidra_out_root,
        denominator_output_path=denominator_output,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
        required_locality_level_id=required_locality_level_id,
        use_cache=not no_cache,
    )

    console.print("[green]SIDRA population imported into bundle.[/green]")
    console.print(f"population_field_id: {result.population_field_id}")
    console.print(f"denominator: {result.denominator_path}")
    console.print(f"projection_report: {result.projection_report_path}")
    console.print(f"rows: {result.n_population_rows}")

    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"- {warning}")
            
            

@app.command("local-sim-sidra-mortality")
def local_sim_sidra_mortality(
    raw_sim: Path = typer.Option(
        ...,
        "--raw-sim",
        help="Raw/local SIM-DO CSV or Parquet file.",
    ),
    sidra_population_spec: Path = typer.Option(
        ...,
        "--sidra-population-spec",
        help="SIDRA population query YAML/JSON spec.",
    ),
    sidra_out_root: Path = typer.Option(
        ...,
        "--sidra-out-root",
        help="SIDRA raw/facts output directory.",
    ),
    sidra_denominator_output: Path = typer.Option(
        ...,
        "--sidra-denominator-output",
        help="SIDRA-derived population denominator Parquet.",
    ),
    intent: Path = typer.Option(
        Path("config/intents/alagoas_smoke.json"),
        "--intent",
        help="Intent JSON file.",
    ),
    processed_sim: Path | None = typer.Option(
        None,
        "--processed-sim",
        help="Optional processed SIM-DO CSV or Parquet file.",
    ),
    normalization_input_kind: str = typer.Option(
        "raw",
        "--normalization-input",
        help="raw or processed.",
    ),
    geography: str = typer.Option(
        "residence",
        "--geography",
        help="residence or occurrence.",
    ),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    sidra_table_id: str | None = typer.Option(None, "--sidra-table-id"),
    sidra_variable_id: str | None = typer.Option(None, "--sidra-variable-id"),
    population_source_label: str = typer.Option(
        "SIDRA_population",
        "--population-source-label",
        help="Label for SIDRA population denominator.",
    ),
    required_locality_level_id: str | None = typer.Option(
        "6",
        "--required-locality-level-id",
        help="SIDRA territorial level code required for denominator export.",
    ),
    no_sidra_cache: bool = typer.Option(
        False,
        "--no-sidra-cache",
        help="Force SIDRA HTTP request instead of using raw JSON cache.",
    ),
) -> None:
    """Run SIM-DO local file + SIDRA population → crude mortality vertical slice."""
    if normalization_input_kind not in {"raw", "processed"}:
        raise typer.BadParameter("normalization-input must be raw or processed.")

    if geography not in {"residence", "occurrence"}:
        raise typer.BadParameter("geography must be residence or occurrence.")

    result = run_local_sim_sidra_mortality(
        root=root,
        intent_path=intent,
        raw_sim_path=raw_sim,
        processed_sim_path=processed_sim,
        sidra_population_spec_path=sidra_population_spec,
        sidra_out_root=sidra_out_root,
        sidra_denominator_output_path=sidra_denominator_output,
        normalization_input_kind=normalization_input_kind,  # type: ignore[arg-type]
        geography=geography,  # type: ignore[arg-type]
        sidra_table_id=sidra_table_id,
        sidra_variable_id=sidra_variable_id,
        population_source_label=population_source_label,
        required_locality_level_id=required_locality_level_id,
        use_sidra_cache=not no_sidra_cache,
    )

    console.print("[green]Local SIM + SIDRA mortality run complete.[/green]")
    console.print(f"run_dir: {result.run_dir}")
    console.print(f"normalized_sim: {result.normalized_sim_path}")
    console.print(f"sidra_denominator: {result.sidra_denominator_path}")
    console.print(f"sidra_projection_report: {result.sidra_projection_report_path}")
    console.print(f"death_count_field_id: {result.death_count_field_id}")
    console.print(f"population_field_id: {result.population_field_id}")
    console.print(f"crude_mortality_field_id: {result.crude_mortality_field_id}")
    console.print(f"summary: {result.summary_path}")
    
    
    
    
@app.command("datasus-ingest")
def datasus_ingest(
    system: str = typer.Option(..., "--system", help="SIM-DO, SINASC, SIH-RD, or CNES-ST."),
    uf: str = typer.Option(..., "--uf", help="UF code, e.g. AL."),
    year_start: int = typer.Option(..., "--year-start"),
    year_end: int | None = typer.Option(None, "--year-end"),
    month_start: int | None = typer.Option(None, "--month-start"),
    month_end: int | None = typer.Option(None, "--month-end"),
    root: Path = typer.Option(Path("."), "--root", help="Project root."),
    rscript: str = typer.Option("Rscript", "--rscript", help="Rscript executable path."),
    no_cache: bool = typer.Option(False, "--no-cache"),
    timeout_seconds: int = typer.Option(7200, "--timeout-seconds"),
    heartbeat_timeout_seconds: int = typer.Option(900, "--heartbeat-timeout-seconds"),
) -> None:
    """Fetch/process DATASUS through R microdatasus and materialize raw/processed Parquet."""
    from pegasus.datasus.client_microdatasus import normalize_datasus_system

    normalized_system = normalize_datasus_system(system)

    request = MicrodatasusRequest(
        system=normalized_system,  # type: ignore[arg-type]
        uf=uf,
        year_start=year_start,
        year_end=year_start if year_end is None else year_end,
        month_start=month_start,
        month_end=month_end,
    )

    manifest = ingest_microdatasus(
        root=root,
        request=request,
        rscript_path=rscript,
        timeout_seconds=timeout_seconds,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        use_cache=not no_cache,
    )

    if manifest.status == "ok":
        console.print("[green]microdatasus ingestion complete.[/green]")
    elif manifest.status == "raw_ok_process_error":
        console.print("[yellow]microdatasus raw fetch succeeded; processing failed.[/yellow]")
    else:
        console.print("[red]microdatasus ingestion failed.[/red]")

    console.print(f"system: {manifest.request.system}")
    console.print(f"uf: {manifest.request.uf}")
    console.print(f"period: {manifest.request.period_label()}")
    console.print(f"request_hash: {manifest.request_hash}")
    console.print(f"status: {manifest.status}")
    console.print(f"manifest: {manifest.paths.final_manifest_json}")
    console.print(f"raw_parquet: {manifest.paths.raw_parquet}")
    console.print(f"processed_parquet: {manifest.paths.processed_parquet}")
    console.print(f"raw_profile: {manifest.paths.raw_profile_json}")
    console.print(f"processed_profile: {manifest.paths.processed_profile_json}")
    console.print(f"comparison: {manifest.paths.comparison_json}")
    console.print(f"r_stdout: {manifest.paths.stdout_log}")
    console.print(f"r_stderr: {manifest.paths.stderr_log}")
    console.print(f"r_work_dir: {manifest.paths.work_dir}")

    if manifest.elapsed_seconds is not None:
        console.print(f"elapsed_seconds: {manifest.elapsed_seconds}")

    if manifest.raw_rows is not None:
        console.print(f"raw_rows: {manifest.raw_rows}")

    if manifest.processed_rows is not None:
        console.print(f"processed_rows: {manifest.processed_rows}")

    if manifest.error_message:
        console.print(f"[red]error:[/red] {manifest.error_message}")

    if manifest.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in manifest.warnings:
            console.print(f"- {warning}")