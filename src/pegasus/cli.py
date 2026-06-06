from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import typer
from rich import print

from pegasus.core.config import validate_config_tree
from pegasus.core.paths import ensure_data_lake
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.validators import validate_registry_tree

app = typer.Typer(no_args_is_help=True)
registries_app = typer.Typer(no_args_is_help=True)
sidra_app = typer.Typer(no_args_is_help=True)
datasus_app = typer.Typer(no_args_is_help=True)

app.add_typer(registries_app, name="registries")
app.add_typer(sidra_app, name="sidra")
app.add_typer(datasus_app, name="datasus")


def _fail(errors: list[str]) -> None:
    for error in errors:
        print(f"[red]ERROR[/red] {error}")
    raise typer.Exit(1)


@app.command()
def init() -> None:
    ensure_data_lake(".")
    run_dir = create_empty_output_bundle(Path("data/runs/slice0_empty"))
    print(f"[green]initialized[/green] data lake and scaffold run: {run_dir}")


@app.command("validate-config")
def validate_config() -> None:
    errors = validate_config_tree(".")
    if errors:
        _fail(errors)
    print("[green]config valid[/green]")


@registries_app.command("validate")
def validate_registries() -> None:
    errors = validate_registry_tree("config/registries")
    if errors:
        _fail(errors)
    print("[green]registries valid[/green]")


@app.command("validate-run")
def validate_run(run: Path = typer.Option(..., "--run")) -> None:
    result = validate_output_bundle(run_dir=str(run))
    if not result.ok:
        _fail(result.errors)
    print("[green]run bundle valid[/green]")


@app.command()
def doctor() -> None:
    checks: dict[str, str] = {}
    checks["python"] = sys.version.split()[0]

    for mod in ["duckdb", "polars", "pyarrow", "pydantic", "typer", "yaml"]:
        checks[mod] = "ok" if importlib.util.find_spec(mod) else "missing"

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec:
        import torch
        checks["torch"] = getattr(torch, "__version__", "ok")
        checks["torch_cuda_available"] = str(torch.cuda.is_available())
        checks["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    else:
        checks["torch"] = "missing"
        checks["torch_cuda_available"] = "False"
        checks["cuda_device_name"] = "none"

    checks["Rscript"] = shutil.which("Rscript") or "missing"
    checks["data_write"] = "ok"
    try:
        ensure_data_lake(".")
    except Exception as exc:
        checks["data_write"] = f"failed: {exc}"

    reg_errors = validate_registry_tree("config/registries")
    checks["registry_schema"] = "ok" if not reg_errors else f"{len(reg_errors)} errors"

    for key, value in checks.items():
        color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
        print(f"[{color}]{key}[/] {value}")

    print("[yellow]doctor is Slice 0-light: no DATASUS/SIDRA ingestion is executed.[/yellow]")


@sidra_app.command("metadata")
def sidra_metadata(tables: Path = typer.Option(..., "--tables")) -> None:
    print(f"[yellow]blocked[/yellow] SIDRA metadata is Slice 2. Seed received: {tables}")
    raise typer.Exit(2)


@sidra_app.command("plan")
def sidra_plan(view: str = typer.Option(..., "--view")) -> None:
    print(f"[yellow]blocked[/yellow] SIDRA planning is Slice 2. View received: {view}")
    raise typer.Exit(2)


@sidra_app.command("extract")
def sidra_extract(plan: Path = typer.Option(..., "--plan")) -> None:
    print(f"[yellow]blocked[/yellow] SIDRA extraction is Slice 2. Plan received: {plan}")
    raise typer.Exit(2)


@datasus_app.command("ingest")
def datasus_ingest(
    system: str = typer.Option(..., "--system"),
    uf: str = typer.Option(..., "--uf"),
    years: str = typer.Option(..., "--years"),
) -> None:
    print(f"[yellow]blocked[/yellow] DATASUS ingest is Slice 1+. Request: {system} {uf} {years}")
    raise typer.Exit(2)


@datasus_app.command("profile")
def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
    print(f"[yellow]blocked[/yellow] DATASUS profiling is Slice 1+. Manifest: {manifest}")
    raise typer.Exit(2)


@app.command()
def compile(intent: Path = typer.Option(..., "--intent")) -> None:
    print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
    raise typer.Exit(2)
