from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import typer
from rich import print

from pegasus.core.config import validate_config_tree
from pegasus.core.paths import ensure_data_lake
from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.manifests import (
    build_datasus_manifests,
    load_datasus_config,
    read_request_manifest,
    write_request_manifest,
)
from pegasus.datasus.profile import profile_table
from pegasus.datasus.schema_compare import compare_profiles
from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
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
    checks["microdatasus"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"
    checks["read.dbc"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"

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

    print("[yellow]doctor is still light: no DATASUS/SIDRA ingestion is executed.[/yellow]")


@datasus_app.command("ingest")
def datasus_ingest(
    system: str = typer.Option(..., "--system"),
    uf: str = typer.Option(..., "--uf"),
    years: str = typer.Option(..., "--years"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan and persist manifests without invoking R."),
) -> None:
    cfg_payload = load_datasus_config()
    cfg = DatasusConfig.from_mapping(cfg_payload)
    manifests = build_datasus_manifests(system=system, uf=uf, years=years, config=cfg_payload)

    cache = DatasusCache()
    blocked = False
    failed = False

    for manifest in manifests:
        planned_path = write_request_manifest(manifest)
        print(f"[cyan]planned[/cyan] {manifest.system} {manifest.uf} {manifest.year_start}: {planned_path}")

        if dry_run:
            continue

        executed = fetch_datasus_chunk(
            manifest,
            config=cfg,
            cache=cache,
            timeout_seconds=cfg.r_timeout_seconds,
            heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
        )
        executed_path = write_request_manifest(executed)

        if executed.status == "success":
            print(f"[green]success[/green] {executed.system} {executed.uf} {executed.year_start}: {executed_path}")
        elif executed.status == "blocked":
            blocked = True
            print(f"[yellow]blocked[/yellow] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")
        else:
            failed = True
            print(f"[red]{executed.status}[/red] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")

    if failed:
        raise typer.Exit(1)
    if blocked:
        raise typer.Exit(2)


@datasus_app.command("profile")
def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
    request = read_request_manifest(manifest)

    raw_path = Path(request.raw_path)
    processed_path = Path(request.processed_path)

    if not raw_path.exists() or not processed_path.exists():
        print("[yellow]blocked[/yellow] raw/processed artifacts are missing; cannot profile this manifest.")
        raise typer.Exit(2)

    raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
    processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
    compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

    raw_profile = profile_table(raw_path, output_path=raw_profile_path)
    processed_profile = profile_table(processed_path, output_path=processed_profile_path)
    compare_profiles(raw_profile, processed_profile, output_path=compare_path)

    print(f"[green]raw profile[/green] {raw_profile_path}")
    print(f"[green]processed profile[/green] {processed_profile_path}")
    print(f"[green]schema comparison[/green] {compare_path}")


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


@app.command()
def compile(intent: Path = typer.Option(..., "--intent")) -> None:
    print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
    raise typer.Exit(2)
