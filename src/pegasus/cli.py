from __future__ import annotations

import importlib.util
import shutil
import sys
import json
from pathlib import Path

import typer
from rich import print

from pegasus.core.config import validate_config_tree
from pegasus.core.paths import ensure_data_lake
from pegasus.output.schema_seed import create_schema_seed_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.validators import validate_registry_tree
from pegasus.workflows.compile import run_compile
from pegasus.sidra.api import SidraClient, SidraClientConfig
from pegasus.workflows.datasus import run_datasus_ingest, run_datasus_normalize_sim, run_datasus_profile
from pegasus.workflows.efg import run_attach_sidra_denominator
from pegasus.workflows.sinasc import run_datasus_normalize_sinasc
from pegasus.workflows.sidra import (
    run_sidra_extract,
    run_sidra_metadata,
    run_sidra_metadata_development,
    run_sidra_normalize_development,
    run_sidra_plan,
    run_sidra_plan_development,
    sidra_runtime_config,
)

app = typer.Typer(no_args_is_help=True)
registries_app = typer.Typer(no_args_is_help=True)
sidra_app = typer.Typer(no_args_is_help=True)
datasus_app = typer.Typer(no_args_is_help=True)
efg_app = typer.Typer(no_args_is_help=True)
population_app = typer.Typer(no_args_is_help=True)
pirs_app = typer.Typer(no_args_is_help=True)
acceptance_app = typer.Typer(no_args_is_help=True)
source_artifacts_app = typer.Typer(no_args_is_help=True)

app.add_typer(registries_app, name="registries")
app.add_typer(sidra_app, name="sidra")
app.add_typer(datasus_app, name="datasus")
app.add_typer(efg_app, name="efg")
app.add_typer(population_app, name="population")
app.add_typer(pirs_app, name="pirs")
app.add_typer(acceptance_app, name="acceptance")
app.add_typer(source_artifacts_app, name="source-artifacts")


def _fail(errors: list[str]) -> None:
    for error in errors:
        print(f"[red]ERROR[/red] {error}")
    raise typer.Exit(1)


@app.command()
def init() -> None:
    ensure_data_lake(".")
    run_dir = create_schema_seed_output_bundle(Path("data/runs/schema_seed_empty"))
    print(f"[green]initialized[/green] data lake and schema-seeded run: {run_dir}")


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
    source_manifest_hash: str = typer.Option("development", "--source-manifest-hash"),
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
    source_manifest_hash: str = typer.Option("development", "--source-manifest-hash"),
) -> None:
    result = run_datasus_normalize_sinasc(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
    print(f"[green]sinasc normalized[/green] rows={result['row_count']} output={result['output_path']}")


@efg_app.command("build-sim-development")
def efg_build_sim_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def efg_build_sinasc_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


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


@sidra_app.command("metadata-development")
def sidra_metadata_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def sidra_plan_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def sidra_normalize_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


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
    source_manifest: Path | None = typer.Option(None, "--source-manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
    try:
        result = run_compile(
            intent_path=intent,
            run_dir=run_dir,
            source_manifest=source_manifest,
            require_materialized_external=require_materialized_external,
        )
    except ValueError as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]compile complete[/green] run={result['run_dir']}")


@efg_app.command("attach-race-bridge")
def efg_attach_race_bridge(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


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


@datasus_app.command("normalize-cnes")
def datasus_normalize_cnes(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("development", "--source-manifest-hash"),
) -> None:
    from pegasus.workflows.cnes_sih import run_datasus_normalize_cnes
    result = run_datasus_normalize_cnes(input_path=input_path, output_path=output_path, source_manifest_hash=source_manifest_hash)
    print(f"[green]cnes normalized[/green] rows={result['row_count']} output={result['output_path']}")


@datasus_app.command("normalize-sih")
def datasus_normalize_sih(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("development", "--source-manifest-hash"),
) -> None:
    from pegasus.workflows.cnes_sih import run_datasus_normalize_sih
    result = run_datasus_normalize_sih(input_path=input_path, output_path=output_path, source_manifest_hash=source_manifest_hash)
    print(f"[green]sih normalized[/green] rows={result['row_count']} output={result['output_path']}")


@efg_app.command("build-cnes-sih-development")
def efg_build_cnes_sih_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def population_build_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def population_plan_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def sidra_context_plan_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def sidra_context_build_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def pirs_plan_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def pirs_build_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def pirs_hsic_plan_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def pirs_hsic_build_development(*args, **kwargs) -> None:
    print("[red]ERROR[/red] retired development/manual command is not available in the production CLI.")
    raise typer.Exit(2)


def dashboard_assert_read_only() -> None:
    from pegasus.dashboard.contracts import dashboard_policy_manifest

    typer.echo(json.dumps(dashboard_policy_manifest(), indent=2, sort_keys=True))


@dashboard_app.command("inspect-run")
def dashboard_inspect_run(
    run: Path = typer.Option(..., "--run"),
) -> None:
    from pegasus.workflows.dashboard import run_dashboard_inspect_run

    typer.echo(json.dumps(run_dashboard_inspect_run(run_dir=run), indent=2, sort_keys=True))


@dashboard_app.command("table-head")
def dashboard_table_head(
    run: Path = typer.Option(..., "--run"),
    table: str = typer.Option(..., "--table"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    from pegasus.workflows.dashboard import run_dashboard_table_head

    typer.echo(json.dumps(run_dashboard_table_head(run_dir=run, table_name=table, limit=limit), indent=2, sort_keys=True))

# Slice 11A acceptance hardening commands
@acceptance_app.command("plan")
def acceptance_plan() -> None:
    from pegasus.workflows.acceptance import run_acceptance_plan

    typer.echo(json.dumps(run_acceptance_plan(), indent=2, sort_keys=True))


@acceptance_app.command("check-run")
def acceptance_check_run(
    run: Path = typer.Option(..., "--run"),
    require_nonempty: bool = typer.Option(False, "--require-nonempty"),
) -> None:
    from pegasus.workflows.acceptance import run_acceptance_check_run

    result = run_acceptance_check_run(run_dir=run, require_nonempty=require_nonempty)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok", False):
        raise typer.Exit(1)



@acceptance_app.command("level3")
def acceptance_level3(
    run: Path = typer.Option(..., "--run"),
) -> None:
    from pegasus.workflows.acceptance import run_acceptance_level3

    result = run_acceptance_level3(run_dir=run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok", False):
        raise typer.Exit(1)


# Slice 12A source artifact reality gate commands
@source_artifacts_app.command("inspect")
def source_artifacts_inspect(
    path: Path = typer.Option(..., "--path"),
    source_system: str = typer.Option(..., "--source-system"),
    artifact_role: str = typer.Option(..., "--role"),
    provenance_mode: str = typer.Option("materialized_external", "--provenance-mode"),
    output: Path | None = typer.Option(None, "--output"),
    source_manifest_hash: str | None = typer.Option(None, "--source-manifest-hash"),
) -> None:
    from pegasus.workflows.source_artifacts import run_source_artifact_inspect

    result = run_source_artifact_inspect(
        path=path,
        source_system=source_system,
        artifact_role=artifact_role,
        provenance_mode=provenance_mode,
        output=output,
        source_manifest_hash=source_manifest_hash,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@source_artifacts_app.command("validate-manifest")
def source_artifacts_validate_manifest(
    manifest: Path = typer.Option(..., "--manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
    from pegasus.workflows.source_artifacts import run_source_manifest_validate

    result = run_source_manifest_validate(
        manifest=manifest,
        require_materialized_external=require_materialized_external,
    )
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise typer.Exit(1)


@source_artifacts_app.command("summary")
def source_artifacts_summary(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    from pegasus.workflows.source_artifacts import run_source_manifest_summary

    typer.echo(json.dumps(run_source_manifest_summary(manifest=manifest), indent=2, sort_keys=True))


@source_artifacts_app.command("compile-reality-plan")
def source_artifacts_compile_reality_plan(
    source_manifest: Path | None = typer.Option(None, "--source-manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
    from pegasus.workflows.compile_source import run_compile_source_reality_plan

    typer.echo(json.dumps(
        run_compile_source_reality_plan(
            source_manifest=source_manifest,
            require_materialized_external=require_materialized_external,
        ),
        indent=2,
        sort_keys=True,
    ))


# ---- Slice 13A SHE substrate CLI ----
she_app = typer.Typer(help="Substrate Harmonization Engine inspection commands.")
app.add_typer(she_app, name="she")


@she_app.command("build-substrate")
def she_build_substrate(
    artifact: list[Path] = typer.Option([], "--artifact", help="Processed source artifact path; repeatable."),
    source_system: str = typer.Option("UNKNOWN", "--source-system"),
    provenance_mode: str = typer.Option("materialized_external", "--provenance-mode"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    from pegasus.workflows.build_substrate import run_build_substrate_from_artifacts

    artifacts = [
        {
            "path": str(path),
            "source_system": source_system,
            "artifact_role": "processed_events",
            "provenance_mode": provenance_mode,
        }
        for path in (artifact or [])
    ]
    result = run_build_substrate_from_artifacts(artifacts=artifacts, output=output)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@she_app.command("build-substrate-manifest")
def she_build_substrate_manifest(
    source_manifest: Path = typer.Option(..., "--source-manifest"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    from pegasus.workflows.build_substrate import run_build_substrate_from_source_manifest

    result = run_build_substrate_from_source_manifest(source_manifest=source_manifest, output=output)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@she_app.command("substrate-summary")
def she_substrate_summary(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    from pegasus.workflows.build_substrate import run_substrate_summary

    typer.echo(json.dumps(run_substrate_summary(manifest=manifest), indent=2, sort_keys=True))
# ---- End Slice 13A SHE substrate CLI ----

# Slice 13B registry-backed source-field commands
@registries_app.command("source-fields-summary")
def registries_source_fields_summary(
    registry_root: Path = typer.Option(Path("config/registries"), "--registry-root"),
) -> None:
    from pegasus.registries.source_fields import source_field_registry_summary

    typer.echo(json.dumps(source_field_registry_summary(registry_root=registry_root), indent=2, sort_keys=True))


@registries_app.command("source-field-resolve")
def registries_source_field_resolve(
    source_system: str = typer.Option(..., "--source-system"),
    column: str = typer.Option(..., "--column"),
    registry_root: Path = typer.Option(Path("config/registries"), "--registry-root"),
) -> None:
    from pegasus.she.source_registry import resolve_source_field

    result = resolve_source_field(source_system=source_system, column_name=column, registry_root=registry_root)
    typer.echo(json.dumps(result.as_manifest(), indent=2, sort_keys=True))

# ---- Slice 14C EFG materialization CLI boundary ----
@efg_app.command("materialize-substrate-manifest")
def efg_materialize_substrate_manifest(
    substrate_manifest: Path = typer.Option(..., "--substrate-manifest"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Build a metadata-only EFG materialization manifest from a substrate manifest."""
    import json

    from pegasus.workflows.efg_materialize import run_materialize_substrate_manifest

    payload = run_materialize_substrate_manifest(substrate_manifest=substrate_manifest, output=output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


@efg_app.command("attach-materialization")
def efg_attach_materialization_manifest(
    run_dir: Path = typer.Option(..., "--run-dir"),
    substrate_manifest: Path | None = typer.Option(None, "--substrate-manifest"),
) -> None:
    """Attach EFG substrate materialization metadata to an existing run bundle."""
    import json

    from pegasus.workflows.efg_materialize import run_attach_efg_materialization_to_run

    payload = run_attach_efg_materialization_to_run(run_dir=run_dir, substrate_manifest=substrate_manifest)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


@efg_app.command("inspect-materialization")
def efg_inspect_materialization_manifest(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Inspect a metadata-only EFG materialization manifest without mutating a run."""
    import json

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"EFG materialization manifest is not a JSON object: {manifest}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {
            "status": "evaluated",
            "materialization_id": payload.get("materialization_id"),
            "substrate_id": payload.get("substrate_id"),
            "field_count": payload.get("field_count"),
            "excluded_field_count": payload.get("excluded_field_count"),
            "metadata_only": payload.get("metadata_only"),
            "writes_v_fields": payload.get("writes_v_fields"),
            "writes_e_dag": payload.get("writes_e_dag"),
        }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
# ---- End Slice 14C EFG materialization CLI boundary ----

# ---- Slice 15D EFG promotion CLI boundary ----
@efg_app.command("plan-promotion")
def efg_plan_promotion(
    run_dir: Path = typer.Option(..., "--run-dir"),
    materialization_manifest: Path = typer.Option(..., "--materialization-manifest"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Build a non-mutating EFG promotion plan from a materialization manifest."""
    import json

    from pegasus.workflows.efg_promotion import run_plan_efg_promotion

    payload = run_plan_efg_promotion(
        run_dir=run_dir,
        materialization_manifest=materialization_manifest,
        output=output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


@efg_app.command("attach-promotion-plan")
def efg_attach_promotion_plan(
    run_dir: Path = typer.Option(..., "--run-dir"),
    materialization_manifest: Path | None = typer.Option(None, "--materialization-manifest"),
) -> None:
    """Attach a non-mutating EFG promotion plan and gate summary to a run bundle."""
    import json

    from pegasus.workflows.efg_promotion import run_attach_efg_promotion_plan_to_run

    payload = run_attach_efg_promotion_plan_to_run(
        run_dir=run_dir,
        materialization_manifest=materialization_manifest,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


@efg_app.command("apply-promotion-plan")
def efg_apply_promotion_plan(
    run_dir: Path = typer.Option(..., "--run-dir"),
    promotion_plan: Path | None = typer.Option(None, "--promotion-plan"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
) -> None:
    """Apply planned EFG promotions to descriptive/quarantined bundle surfaces."""
    import json

    from pegasus.workflows.efg_apply import run_apply_efg_promotion_plan

    payload = run_apply_efg_promotion_plan(
        run_dir=run_dir,
        promotion_plan=promotion_plan,
        validate=validate,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str))


@efg_app.command("inspect-promotion-plan")
def efg_inspect_promotion_plan(
    plan: Path = typer.Option(..., "--plan"),
) -> None:
    """Inspect an EFG promotion plan without mutating a run bundle."""
    import json

    payload = json.loads(plan.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"EFG promotion plan is not a JSON object: {plan}")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {
            "status": payload.get("status", "evaluated"),
            "promotion_plan_id": payload.get("promotion_plan_id"),
            "planned_promotion_count": payload.get("planned_promotion_count"),
            "conflict_count": payload.get("conflict_count"),
            "excluded_source_field_count": payload.get("excluded_source_field_count"),
            "mutates_v_fields": payload.get("mutates_v_fields", False),
        }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str))
# ---- End Slice 15D EFG promotion CLI boundary ----

# ---- Slice 16E PIRS planning CLI boundary ----
def _pirs_cli_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON artifact: {path}: {exc}") from exc


def _pirs_cli_print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@pirs_app.command("candidates-from-run")
def pirs_candidates_from_run(
    run_dir: Path = typer.Option(..., "--run-dir"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Build a non-mutating PIRS field-candidate manifest from a completed run."""
    from pegasus.workflows.pirs_candidates import run_build_pirs_candidates_from_run

    _pirs_cli_print(run_build_pirs_candidates_from_run(run_dir=run_dir, output=output))


@pirs_app.command("attach-candidate-gate")
def pirs_attach_candidate_gate(
    run_dir: Path = typer.Option(..., "--run-dir"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Write and attach the PIRS candidate-gate summary to an existing run."""
    from pegasus.workflows.pirs_candidates import run_attach_pirs_candidate_gate

    _pirs_cli_print(run_attach_pirs_candidate_gate(run_dir=run_dir, output=output))


@pirs_app.command("plan-selection")
def pirs_plan_selection(
    run_dir: Path = typer.Option(..., "--run-dir"),
    candidate_manifest: Path | None = typer.Option(None, "--candidate-manifest"),
    output: Path | None = typer.Option(None, "--output"),
    budget: str = typer.Option("fast", "--budget"),
) -> None:
    """Build a non-mutating PIRS selection plan from candidate manifest data."""
    from pegasus.workflows.pirs_selection import run_write_pirs_selection_plan

    _pirs_cli_print(
        run_write_pirs_selection_plan(
            run_dir=run_dir,
            candidate_manifest=candidate_manifest,
            output=output,
            budget=budget,
        )
    )


@pirs_app.command("attach-selection-plan")
def pirs_attach_selection_plan(
    run_dir: Path = typer.Option(..., "--run-dir"),
    candidate_manifest: Path | None = typer.Option(None, "--candidate-manifest"),
    output: Path | None = typer.Option(None, "--output"),
    budget: str = typer.Option("fast", "--budget"),
) -> None:
    """Write and attach a PIRS selection-plan summary to an existing run."""
    from pegasus.workflows.pirs_selection import run_attach_pirs_selection_plan

    _pirs_cli_print(
        run_attach_pirs_selection_plan(
            run_dir=run_dir,
            candidate_manifest=candidate_manifest,
            output=output,
            budget=budget,
        )
    )


@pirs_app.command("plan-design")
def pirs_plan_design(
    run_dir: Path = typer.Option(..., "--run-dir"),
    selection_plan: Path | None = typer.Option(None, "--selection-plan"),
    output: Path | None = typer.Option(None, "--output"),
    budget: str | None = typer.Option(None, "--budget"),
) -> None:
    """Build a planned-only PIRS model-design contract from a selection plan."""
    from pegasus.workflows.pirs_design import run_plan_pirs_design

    _pirs_cli_print(
        run_plan_pirs_design(
            run_dir=run_dir,
            selection_plan=selection_plan,
            output=output,
            budget=budget,
        )
    )


@pirs_app.command("attach-design-plan")
def pirs_attach_design_plan(
    run_dir: Path = typer.Option(..., "--run-dir"),
    selection_plan: Path | None = typer.Option(None, "--selection-plan"),
    output: Path | None = typer.Option(None, "--output"),
    budget: str | None = typer.Option(None, "--budget"),
) -> None:
    """Write and attach a PIRS design-plan summary to an existing run."""
    from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run

    _pirs_cli_print(
        run_attach_pirs_design_plan_to_run(
            run_dir=run_dir,
            selection_plan=selection_plan,
            output=output,
            budget=budget,
        )
    )


@pirs_app.command("check-design-readiness")
def pirs_check_design_readiness(
    run_dir: Path = typer.Option(..., "--run-dir"),
    design_plan: Path | None = typer.Option(None, "--design-plan"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Write a design-readiness manifest without attaching it to run metadata."""
    from pegasus.workflows.pirs_readiness import run_write_pirs_design_readiness_manifest

    _pirs_cli_print(
        run_write_pirs_design_readiness_manifest(
            run_dir=run_dir,
            design_plan=design_plan,
            output=output,
        )
    )


@pirs_app.command("attach-design-readiness")
def pirs_attach_design_readiness(
    run_dir: Path = typer.Option(..., "--run-dir"),
    design_plan: Path | None = typer.Option(None, "--design-plan"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Write and attach the PIRS design-readiness gate to an existing run."""
    from pegasus.workflows.pirs_readiness import run_attach_pirs_design_readiness_to_run

    _pirs_cli_print(
        run_attach_pirs_design_readiness_to_run(
            run_dir=run_dir,
            design_plan=design_plan,
            output=output,
        )
    )


@pirs_app.command("inspect-candidates")
def pirs_inspect_candidates(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Read an existing PIRS candidate manifest without mutating a run."""
    payload = _pirs_cli_json(manifest)
    _pirs_cli_print(
        {
            "artifact": payload.get("gate", "pirs_candidate_gate"),
            "candidate_count": payload.get("candidate_count", len(payload.get("candidates", []))),
            "rejected_count": payload.get("rejected_count", len(payload.get("rejected", []))),
            "manifest": str(manifest),
            "read_only": True,
        }
    )


@pirs_app.command("inspect-selection-plan")
def pirs_inspect_selection_plan(
    plan: Path = typer.Option(..., "--plan"),
) -> None:
    """Read an existing PIRS selection plan without mutating a run."""
    payload = _pirs_cli_json(plan)
    _pirs_cli_print(
        {
            "artifact": payload.get("artifact", "pirs_selection_plan"),
            "status": payload.get("status"),
            "budget": payload.get("budget"),
            "selected_outcome_field_id": payload.get("selected_outcome_field_id"),
            "selected_covariate_count": len(payload.get("selected_covariate_field_ids", [])),
            "selection_rejected_count": len(payload.get("selection_rejected", [])),
            "manifest": str(plan),
            "read_only": True,
        }
    )


@pirs_app.command("inspect-design-plan")
def pirs_inspect_design_plan(
    plan: Path = typer.Option(..., "--plan"),
) -> None:
    """Read an existing PIRS design plan without mutating a run."""
    payload = _pirs_cli_json(plan)
    _pirs_cli_print(
        {
            "artifact": payload.get("artifact", "pirs_design_plan"),
            "status": payload.get("status"),
            "design_matrix_state": payload.get("design_matrix_state"),
            "model_fit_state": payload.get("model_fit_state"),
            "residual_state": payload.get("residual_state"),
            "term_count": len(payload.get("terms", [])),
            "manifest": str(plan),
            "read_only": True,
        }
    )


@pirs_app.command("inspect-design-readiness")
def pirs_inspect_design_readiness(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Read an existing PIRS design-readiness manifest without mutating a run."""
    payload = _pirs_cli_json(manifest)
    _pirs_cli_print(
        {
            "artifact": payload.get("artifact", "pirs_design_readiness_gate"),
            "status": payload.get("status"),
            "ready": payload.get("ready"),
            "accepted_field_count": len(payload.get("accepted_fields", [])),
            "rejected_field_count": len(payload.get("rejected_fields", [])),
            "manifest": str(manifest),
            "read_only": True,
        }
    )
# ---- End Slice 16E PIRS planning CLI boundary ----

# ---- Slice 17B PIRS model execution CLI boundary ----
@pirs_app.command("execute-model")
def pirs_execute_model(
    run_dir: Path = typer.Option(..., "--run-dir"),
    design_matrix_manifest: Path | None = typer.Option(None, "--design-matrix-manifest"),
    output_manifest: Path | None = typer.Option(None, "--output-manifest"),
    mutate_output_bundle: bool = typer.Option(True, "--mutate-output-bundle/--no-mutate-output-bundle"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
    attach: bool = typer.Option(True, "--attach/--no-attach"),
) -> None:
    """Execute PIRS model fitting from a ready design matrix and materialize residual artifacts."""
    from pegasus.workflows.pirs_execute import run_execute_pirs_model

    _pirs_cli_print(run_execute_pirs_model(run_dir=run_dir, design_matrix_manifest=design_matrix_manifest, output_manifest=output_manifest, mutate_output_bundle=mutate_output_bundle, validate=validate, attach=attach))


@pirs_app.command("inspect-model-execution")
def pirs_inspect_model_execution(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Read an existing PIRS model-execution manifest without mutating a run."""
    from pegasus.pirs.model_execution import inspect_pirs_model_execution_manifest

    _pirs_cli_print(inspect_pirs_model_execution_manifest(manifest))
# ---- End Slice 17B PIRS model execution CLI boundary ----

# ---- Slice 18A HSIC residual scan CLI boundary ----
@pirs_app.command("scan-residuals-hsic")
def pirs_scan_residuals_hsic(
    run_dir: Path = typer.Option(..., "--run-dir"),
    model_execution_manifest: Path | None = typer.Option(None, "--model-execution-manifest"),
    design_matrix_manifest: Path | None = typer.Option(None, "--design-matrix-manifest"),
    output_manifest: Path | None = typer.Option(None, "--output-manifest"),
    budget: str = typer.Option("fast", "--budget"),
    permutations: int = typer.Option(199, "--permutations"),
    min_support: int = typer.Option(3, "--min-support"),
    mutate_output_bundle: bool = typer.Option(True, "--mutate-output-bundle/--no-mutate-output-bundle"),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
    attach: bool = typer.Option(True, "--attach/--no-attach"),
) -> None:
    """Scan PIRS residuals against design covariates with HSIC and emit hypothesis rows."""
    from pegasus.workflows.hsic_execute import run_execute_hsic_residual_scan

    _pirs_cli_print(run_execute_hsic_residual_scan(run_dir=run_dir, model_execution_manifest=model_execution_manifest, design_matrix_manifest=design_matrix_manifest, output_manifest=output_manifest, budget=budget, permutations=permutations, min_support=min_support, mutate_output_bundle=mutate_output_bundle, validate=validate, attach=attach))


@pirs_app.command("inspect-hsic-scan")
def pirs_inspect_hsic_scan(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    """Read an existing HSIC residual-scan manifest without mutating a run."""
    from pegasus.workflows.hsic_execute import run_inspect_hsic_residual_scan

    _pirs_cli_print(run_inspect_hsic_residual_scan(manifest=manifest))
# ---- End Slice 18A HSIC residual scan CLI boundary ----

# BEGIN SLICE18B HSIC RANKING DASHBOARD CLI
@pirs_app.command("rank-hsic-scan")
def pirs_rank_hsic_scan(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    scan_manifest: Path | None = typer.Option(None, "--scan-manifest", exists=False, file_okay=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", exists=False, file_okay=True, dir_okay=False),
) -> None:
    """Rank HSIC residual-scan results and write read-only dashboard cards."""
    from pegasus.workflows.hsic_rank import run_rank_hsic_residual_scan

    payload = run_rank_hsic_residual_scan(
        run_dir=run_dir,
        scan_manifest=scan_manifest,
        output=output,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@pirs_app.command("inspect-hsic-ranking")
def pirs_inspect_hsic_ranking(
    manifest: Path = typer.Option(..., "--manifest", exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Inspect an existing HSIC ranking manifest without mutating a run."""
    from pegasus.workflows.hsic_rank import inspect_hsic_ranking_manifest

    payload = inspect_hsic_ranking_manifest(manifest)
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@pirs_app.command("inspect-hsic-dashboard")
def pirs_inspect_hsic_dashboard(
    dashboard: Path = typer.Option(..., "--dashboard", exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Inspect read-only HSIC dashboard cards without mutating a run."""
    from pegasus.dashboard.hsic_readonly import inspect_hsic_dashboard_cards

    payload = inspect_hsic_dashboard_cards(dashboard)
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))
# END SLICE18B HSIC RANKING DASHBOARD CLI

# BEGIN SLICE18C HSIC REPORT EXPORT CLI
@pirs_app.command("export-hsic-report")
def pirs_export_hsic_report(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    ranking_manifest: Path | None = typer.Option(None, "--ranking-manifest", exists=False, file_okay=True, dir_okay=False),
    output_dir: Path | None = typer.Option(None, "--output-dir", exists=False, file_okay=False, dir_okay=True),
) -> None:
    """Export a read-only HSIC evidence report bundle from ranked scan artifacts."""
    from pegasus.workflows.hsic_report import run_export_hsic_report

    payload = run_export_hsic_report(
        run_dir=run_dir,
        ranking_manifest=ranking_manifest,
        output_dir=output_dir,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@pirs_app.command("inspect-hsic-report")
def pirs_inspect_hsic_report(
    manifest: Path = typer.Option(..., "--manifest", exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Inspect an exported HSIC evidence report manifest without mutating a run."""
    from pegasus.workflows.hsic_report import inspect_hsic_report_manifest

    payload = inspect_hsic_report_manifest(manifest)
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))
# END SLICE18C HSIC REPORT EXPORT CLI
