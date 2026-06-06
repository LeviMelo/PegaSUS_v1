from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import typer
from rich import print

from pegasus.core.config import load_yaml, validate_config_tree
from pegasus.core.paths import ensure_data_lake
from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.manifests import (
    build_datasus_manifests,
    load_datasus_config,
    read_request_manifest,
    write_request_manifest,
)
from pegasus.datasus.normalize import normalize_sim_do_events
from pegasus.datasus.profile import profile_table
from pegasus.datasus.schema_compare import compare_profiles
from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.validators import validate_registry_tree
from pegasus.sidra.api import SidraClient, SidraClientConfig
from pegasus.sidra.extract import extract_chunk_plan, read_chunk_plan, write_chunk_plan, write_extraction_log
from pegasus.sidra.facts import normalize_fixture_json_to_facts
from pegasus.sidra.metadata import (
    fetch_official_metadata,
    fixture_sidra_metadata,
    metadata_dir_hash,
    read_normalized_metadata_tables,
    table_ids_from_seed,
    write_normalized_metadata_tables,
)
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.registry import request_from_view
from pegasus.sidra.schemas import SIDRARequest
from pegasus.workflows.build_efg import build_sim_fixture_efg_run

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


def _sidra_runtime_config() -> dict:
    data = load_yaml("config/sidra.yaml")
    return data.get("sidra", data)


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
                    **_sidra_runtime_config(),
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


@datasus_app.command("normalize-sim")
def datasus_normalize_sim(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
) -> None:
    result = normalize_sim_do_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
    print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")


@efg_app.command("build-sim-fixture")
def efg_build_sim_fixture(
    sim_events: Path = typer.Option(..., "--sim-events"),
    run_dir: Path = typer.Option(..., "--run-dir"),
) -> None:
    output = build_sim_fixture_efg_run(
        sim_events_path=sim_events,
        run_dir=run_dir,
    )
    result = validate_output_bundle(run_dir=str(output))
    if not result.ok:
        _fail(result.errors)
    print(f"[green]sim fixture EFG bundle valid[/green] {output}")


@sidra_app.command("metadata-fixture")
def sidra_metadata_fixture(
    output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
) -> None:
    metadata = fixture_sidra_metadata()
    outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)
    for name, path in outputs.items():
        print(f"[green]{name}[/green] {path}")


@sidra_app.command("plan-fixture")
def sidra_plan_fixture(
    output: Path = typer.Option(Path("data/manifests/sidra/fixture_plan.json"), "--output"),
    max_cells: int = typer.Option(49900, "--max-cells"),
) -> None:
    metadata = fixture_sidra_metadata()
    table = metadata.tables["9606"]
    request = SIDRARequest(
        table_id="9606",
        variables=table.variables,
        periods=table.periods,
        locality_level="N6",
        localities=table.localities_by_level["N6"],
        classifications=table.classifications,
    )
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=max_cells)
    write_chunk_plan(chunks, output_path=output)
    print(f"[green]planned[/green] chunks={len(chunks)} output={output}")


@sidra_app.command("normalize-fixture")
def sidra_normalize_fixture(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(Path("data/processed/sidra/facts/9606/fixture.parquet"), "--output"),
) -> None:
    output = normalize_fixture_json_to_facts(
        input_path=input_path,
        output_path=output_path,
        table_id="9606",
        unit_by_variable={"93": "persons"},
    )
    print(f"[green]sidra facts normalized[/green] {output}")


@sidra_app.command("metadata")
def sidra_metadata(
    tables: Path = typer.Option(..., "--tables"),
    level: str = typer.Option("N6", "--level"),
    output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
    raw_dir: Path = typer.Option(Path("data/metadata/sidra/raw"), "--raw-dir"),
) -> None:
    table_ids = table_ids_from_seed(tables)
    if not table_ids:
        print("[red]ERROR[/red] no SIDRA table IDs found in seed.")
        raise typer.Exit(1)

    client = SidraClient()
    metadata = fetch_official_metadata(
        table_ids=table_ids,
        client=client,
        locality_level=level,
        raw_dir=raw_dir,
    )
    outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)

    print(f"[green]metadata rebuilt[/green] tables={len(table_ids)}")
    for name, path in outputs.items():
        print(f"[green]{name}[/green] {path}")


@sidra_app.command("plan")
def sidra_plan(
    view: str = typer.Option(..., "--view"),
    metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    sidra_cfg = _sidra_runtime_config()
    metadata = read_normalized_metadata_tables(metadata_dir)
    request = request_from_view(view)
    chunks = plan_sidra_chunks(
        request,
        metadata,
        max_cells_per_request=int(sidra_cfg.get("max_cells_per_request", 49900)),
    )
    output = output or Path("data/manifests/sidra") / f"{view}.json"
    write_chunk_plan(chunks, output_path=output)

    print(f"[green]planned[/green] view={view} chunks={len(chunks)} output={output}")


@sidra_app.command("extract")
def sidra_extract(
    plan: Path = typer.Option(..., "--plan"),
    metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    concurrency: int | None = typer.Option(None, "--concurrency"),
    log_path: Path | None = typer.Option(None, "--log"),
) -> None:
    sidra_cfg = _sidra_runtime_config()
    chunks = read_chunk_plan(plan)
    metadata_hash = metadata_dir_hash(metadata_dir)

    if dry_run:
        total = sum(c.estimated_cells for c in chunks)
        print(f"[cyan]dry-run[/cyan] chunks={len(chunks)} estimated_cells={total} metadata_hash={metadata_hash}")
        return

    client = SidraClient()
    results = extract_chunk_plan(
        chunks,
        client=client,
        concurrency=concurrency or int(sidra_cfg.get("concurrency", 4)),
        metadata_hash=metadata_hash,
    )
    log_path = log_path or Path("data/diagnostics/sidra") / f"{plan.stem}.extraction_log.json"
    write_extraction_log(results, output_path=log_path)

    failures = [r for r in results if r.status != "success"]
    print(f"[green]extract complete[/green] chunks={len(results)} failures={len(failures)} log={log_path}")
    if failures:
        raise typer.Exit(1)


@app.command()
def compile(intent: Path = typer.Option(..., "--intent")) -> None:
    print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
    raise typer.Exit(2)
