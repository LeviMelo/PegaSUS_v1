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
from pegasus.workflows.compile import compile_run

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


@app.command("compile")
def compile_command(
    intent: Path = typer.Option(..., "--intent", "-i", help="Path to intent JSON."),
    root: Path = typer.Option(Path("."), help="Project root."),
) -> None:
    """Create a PegaSUS run bundle for the provided intent."""
    run_dir = compile_run(intent_path=intent, root=root)
    console.print(f"[green]Run bundle created:[/green] {run_dir}")