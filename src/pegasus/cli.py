from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

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
) -> None:
    """Fetch a SIDRA query, persist raw JSON, normalize facts, and write manifest."""
    manifest = fetch_sidra_query_to_disk(
        root=root,
        spec_path=spec,
        out_root=out_root,
        allow_empty=allow_empty,
    )

    console.print("[green]SIDRA fetch complete.[/green]")
    console.print(f"table_id: {manifest.table_id}")
    console.print(f"periods: {manifest.periods}")
    console.print(f"variables: {manifest.variables}")
    console.print(f"view: {manifest.view}")
    console.print(f"n_raw_top_level_items: {manifest.n_raw_top_level_items}")
    console.print(f"n_facts: {manifest.n_facts}")
    console.print(f"url: {manifest.url}")
    console.print(f"raw_json: {manifest.raw_json_path}")
    console.print(f"facts: {manifest.facts_path}")

    if manifest.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in manifest.warnings:
            console.print(f"- {warning}")