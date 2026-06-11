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
from pegasus.workflows.compile import run_compile
from pegasus.sidra.api import SidraClient, SidraClientConfig
from pegasus.workflows.datasus import run_datasus_ingest, run_datasus_normalize_sim, run_datasus_profile
from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture
from pegasus.workflows.sinasc import run_build_sinasc_fixture, run_datasus_normalize_sinasc
from pegasus.workflows.sidra import (
    run_sidra_extract,
    run_sidra_metadata,
    run_sidra_metadata_fixture,
    run_sidra_normalize_fixture,
    run_sidra_plan,
    run_sidra_plan_fixture,
    sidra_runtime_config,
)

app = typer.Typer(no_args_is_help=True)
registries_app = typer.Typer(no_args_is_help=True)
sidra_app = typer.Typer(no_args_is_help=True)
datasus_app = typer.Typer(no_args_is_help=True)
efg_app = typer.Typer(no_args_is_help=True)

app.add_typer(registries_app, name="registries")
app.add_typer(sidra_app, name="sidra")
app.add_typer(datasus_app, name="datasus")
app.add_typer(efg_app, name="efg")


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

    try:
        client = SidraClient(
            config=SidraClientConfig.from_mapping(
                {
                    **sidra_runtime_config(),
                    "timeout_seconds": 5,
                    "max_retries": 0,
                }
            )
        )
        response = client.ping()
        checks["sidra_network"] = "ok" if response.status_code < 400 else f"HTTP {response.status_code}"
    except Exception as exc:
        checks["sidra_network"] = f"failed: {exc}"

    for key, value in checks.items():
        color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
        print(f"[{color}]{key}[/] {value}")

    print("[yellow]doctor is light: it checks connectivity but does not run heavy ingestion.[/yellow]")


@datasus_app.command("ingest")
def datasus_ingest(
    system: str = typer.Option(..., "--system"),
    uf: str = typer.Option(..., "--uf"),
    years: str = typer.Option(..., "--years"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan and persist manifests without invoking R."),
) -> None:
    result = run_datasus_ingest(system=system, uf=uf, years=years, dry_run=dry_run)
    for manifest, planned_path in result["planned"]:
        print(f"[cyan]planned[/cyan] {manifest.system} {manifest.uf} {manifest.year_start}: {planned_path}")
    for manifest, executed_path in result["executed"]:
        if manifest.status == "success":
            print(f"[green]success[/green] {manifest.system} {manifest.uf} {manifest.year_start}: {executed_path}")
        elif manifest.status == "blocked":
            print(f"[yellow]blocked[/yellow] {manifest.system} {manifest.uf} {manifest.year_start}: {manifest.error_message}")
        else:
            print(f"[red]{manifest.status}[/red] {manifest.system} {manifest.uf} {manifest.year_start}: {manifest.error_message}")
    if result["failed"]:
        raise typer.Exit(1)
    if result["blocked"]:
        raise typer.Exit(2)


@datasus_app.command("profile")
def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
    result = run_datasus_profile(manifest=manifest)
    if result["status"] == "blocked":
        print("[yellow]blocked[/yellow] raw/processed artifacts are missing; cannot profile this manifest.")
        raise typer.Exit(2)
    print(f"[green]raw profile[/green] {result['raw_profile_path']}")
    print(f"[green]processed profile[/green] {result['processed_profile_path']}")
    print(f"[green]schema comparison[/green] {result['compare_path']}")


@datasus_app.command("normalize-sim")
def datasus_normalize_sim(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
) -> None:
    result = run_datasus_normalize_sim(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
    print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")


@datasus_app.command("normalize-sinasc")
def datasus_normalize_sinasc(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
) -> None:
    result = run_datasus_normalize_sinasc(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
    print(f"[green]sinasc normalized[/green] rows={result['row_count']} output={result['output_path']}")


@efg_app.command("build-sim-fixture")
def efg_build_sim_fixture(
    sim_events: Path = typer.Option(..., "--sim-events"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    result = run_build_sim_fixture(
        sim_events_path=sim_events,
        run_dir=run_dir,
        municipality_cod6=municipality_cod6,
    )
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]sim fixture EFG bundle valid[/green] {result['run_dir']}")


@efg_app.command("build-sinasc-fixture")
def efg_build_sinasc_fixture(
    sinasc_events: Path = typer.Option(..., "--sinasc-events"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    result = run_build_sinasc_fixture(
        sinasc_events_path=sinasc_events,
        run_dir=run_dir,
        municipality_cod6=municipality_cod6,
    )
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]SINASC maternal-child EFG bundle valid[/green] {result['run_dir']}")


@efg_app.command("attach-sidra-denominator")
def efg_attach_sidra_denominator(
    run_dir: Path = typer.Option(..., "--run-dir"),
    sidra_facts: Path = typer.Option(..., "--sidra-facts"),
) -> None:
    result = run_attach_sidra_denominator(
        run_dir=run_dir,
        sidra_facts_path=sidra_facts,
    )
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]SIDRA denominator anchor attached and run bundle valid[/green] {result['run_dir']}")


@sidra_app.command("metadata-fixture")
def sidra_metadata_fixture(
    output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
) -> None:
    outputs = run_sidra_metadata_fixture(output_dir=output_dir)
    for name, path in outputs.items():
        print(f"[green]{name}[/green] {path}")


@sidra_app.command("plan-fixture")
def sidra_plan_fixture(
    output: Path = typer.Option(Path("data/manifests/sidra/fixture_plan.json"), "--output"),
    max_cells: int = typer.Option(49900, "--max-cells"),
) -> None:
    result = run_sidra_plan_fixture(output=output, max_cells=max_cells)
    print(f"[green]planned[/green] chunks={len(result['chunks'])} output={result['output']}")


@sidra_app.command("normalize-fixture")
def sidra_normalize_fixture(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(Path("data/processed/sidra/facts/9606/fixture.parquet"), "--output"),
) -> None:
    output = run_sidra_normalize_fixture(input_path=input_path, output_path=output_path)
    print(f"[green]sidra facts normalized[/green] {output}")


@sidra_app.command("metadata")
def sidra_metadata(
    tables: Path = typer.Option(..., "--tables"),
    level: str = typer.Option("N6", "--level"),
    output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
    raw_dir: Path = typer.Option(Path("data/metadata/sidra/raw"), "--raw-dir"),
) -> None:
    try:
        result = run_sidra_metadata(tables=tables, level=level, output_dir=output_dir, raw_dir=raw_dir)
    except ValueError as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    print(f"[green]metadata rebuilt[/green] tables={len(result['table_ids'])}")
    for name, path in result["outputs"].items():
        print(f"[green]{name}[/green] {path}")


@sidra_app.command("plan")
def sidra_plan(
    view: str = typer.Option(..., "--view"),
    metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    result = run_sidra_plan(view=view, metadata_dir=metadata_dir, output=output)
    print(f"[green]planned[/green] view={view} chunks={len(result['chunks'])} output={result['output']}")


@sidra_app.command("extract")
def sidra_extract(
    plan: Path = typer.Option(..., "--plan"),
    metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    concurrency: int | None = typer.Option(None, "--concurrency"),
    log_path: Path | None = typer.Option(None, "--log"),
) -> None:
    result = run_sidra_extract(
        plan=plan,
        metadata_dir=metadata_dir,
        dry_run=dry_run,
        concurrency=concurrency,
        log_path=log_path,
    )
    if result["status"] == "dry_run":
        print(
            f"[cyan]dry-run[/cyan] chunks={len(result['chunks'])} "
            f"estimated_cells={result['estimated_cells']} metadata_hash={result['metadata_hash']}"
        )
        return
    failures = result["failures"]
    print(f"[green]extract complete[/green] chunks={len(result['results'])} failures={len(failures)} log={result['log_path']}")
    if failures:
        raise typer.Exit(1)


@app.command()
def compile(
    intent: Path = typer.Option(..., "--intent"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
) -> None:
    try:
        result = run_compile(intent_path=intent, run_dir=run_dir)
    except ValueError as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]compile complete[/green] run={result['run_dir']}")


@efg_app.command("attach-race-bridge")
def efg_attach_race_bridge(
    run_dir: Path = typer.Option(..., "--run-dir"),
    sim_events: Path = typer.Option(..., "--sim-events"),
    bridge_prior: Path = typer.Option(..., "--bridge-prior"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    from pegasus.workflows.race_bridge import run_attach_race_bridge

    result = run_attach_race_bridge(
        run_dir=run_dir,
        sim_events_path=sim_events,
        bridge_prior_path=bridge_prior,
        municipality_cod6=municipality_cod6,
    )
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]race bridge fields attached and run bundle valid[/green] {result['run_dir']}")


@efg_app.command("validate-race-bridge-prior")
def efg_validate_race_bridge_prior(
    bridge_prior: Path = typer.Option(..., "--bridge-prior"),
) -> None:
    from pegasus.workflows.race_bridge import run_validate_race_bridge_prior

    result = run_validate_race_bridge_prior(bridge_prior_path=bridge_prior)
    print(f"[green]race bridge prior valid[/green] bridge_id={result['bridge_id']} hash={result['prior_hash']}")


@efg_app.command("plan-race-bridge")
def efg_plan_race_bridge(
    sim_events: Path = typer.Option(..., "--sim-events"),
    bridge_prior: Path | None = typer.Option(None, "--bridge-prior"),
    intent: Path | None = typer.Option(None, "--intent"),
    registry: Path = typer.Option(Path("config/registries/race_bridge_priors.yaml"), "--registry"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    from pegasus.workflows.race_bridge import run_plan_race_bridge

    result = run_plan_race_bridge(
        sim_events_path=sim_events,
        bridge_prior_path=bridge_prior,
        intent_path=intent,
        registry_path=registry,
        municipality_cod6=municipality_cod6,
    )
    print(result["summary_json"])
