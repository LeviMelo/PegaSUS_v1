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