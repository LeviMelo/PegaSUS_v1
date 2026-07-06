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
from pegasus.workflows.acquire.datasus import run_datasus_ingest, run_datasus_normalize_sim, run_datasus_profile
from pegasus.workflows.acquire.sinasc import run_datasus_normalize_sinasc
from pegasus.workflows.acquire.sidra import (
    run_sidra_extract,
    run_sidra_metadata,
    run_sidra_plan,
    sidra_runtime_config,
)

app = typer.Typer(no_args_is_help=True)
registries_app = typer.Typer(no_args_is_help=True)
sidra_app = typer.Typer(no_args_is_help=True)
datasus_app = typer.Typer(no_args_is_help=True)
efg_app = typer.Typer(no_args_is_help=True)
population_app = typer.Typer(no_args_is_help=True)
acceptance_app = typer.Typer(no_args_is_help=True)
source_artifacts_app = typer.Typer(no_args_is_help=True)
dashboard_app = typer.Typer(no_args_is_help=True)

app.add_typer(registries_app, name="registries")
app.add_typer(sidra_app, name="sidra")
app.add_typer(datasus_app, name="datasus")
app.add_typer(efg_app, name="efg")
app.add_typer(population_app, name="population")
app.add_typer(acceptance_app, name="acceptance")
app.add_typer(source_artifacts_app, name="source-artifacts")
app.add_typer(dashboard_app, name="dashboard")


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
    source_manifest: Path = typer.Option(..., "--source-manifest"),
    require_materialized_external: bool = typer.Option(True, "--require-materialized-external"),
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


@app.command("run")
def run_live(
    intent: Path = typer.Option(..., "--intent"),
    data_root: Path = typer.Option(Path("data"), "--data-root"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    sidra_metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--sidra-metadata-dir"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve acquisition params from intent without touching R/network."),
) -> None:
    """End-to-end live run: acquire DATASUS + SIDRA from intent, merge one source
    manifest, and compile to an immutable run bundle. No fixtures."""
    import json as _json

    from pegasus.workflows.pipeline import LivePipelineError, plan_live_pipeline, run_live_pipeline

    if dry_run:
        try:
            plan = plan_live_pipeline(intent_path=intent)
        except (LivePipelineError, ValueError) as exc:
            print(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(1) from exc
        print(_json.dumps(plan, indent=2, sort_keys=True))
        return

    try:
        result = run_live_pipeline(
            intent_path=intent,
            data_root=data_root,
            run_dir=run_dir,
            sidra_metadata_dir=sidra_metadata_dir,
        )
    except (LivePipelineError, ValueError) as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    for artifact in result.datasus_artifacts:
        print(f"[cyan]datasus[/cyan] {artifact.get('source_system')} rows={artifact.get('row_count')} {artifact.get('path')}")
    if result.sidra_artifact:
        print(f"[cyan]sidra[/cyan] population rows={result.sidra_artifact.get('row_count')} {result.sidra_artifact.get('path')}")
    if result.status != "success":
        print(f"[red]pipeline {result.status}[/red] {result.reason}")
        raise typer.Exit(1)
    print(f"[green]live pipeline complete[/green] run={result.run_dir} manifest={result.source_manifest}")


def efg_validate_race_bridge_prior(
    bridge_prior: Path = typer.Option(..., "--bridge-prior"),
) -> None:
    from pegasus.workflows.report.race_bridge import run_validate_race_bridge_prior

    result = run_validate_race_bridge_prior(bridge_prior_path=bridge_prior)
    print(f"[green]race bridge prior valid[/green] bridge_id={result['bridge_id']} hash={result['prior_hash']}")


@efg_app.command("plan-race-bridge")
def efg_plan_race_bridge(
    sim_events: Path = typer.Option(..., "--sim-events"),
    bridge_prior: Path | None = typer.Option(None, "--bridge-prior"),
    intent: Path | None = typer.Option(None, "--intent"),
    registry: Path = typer.Option(Path("config/registries/demographic/race_bridge_priors.yaml"), "--registry"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    from pegasus.workflows.report.race_bridge import run_plan_race_bridge

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
    from pegasus.workflows.acquire.cnes_sih import run_datasus_normalize_cnes
    result = run_datasus_normalize_cnes(input_path=input_path, output_path=output_path, source_manifest_hash=source_manifest_hash)
    print(f"[green]cnes normalized[/green] rows={result['row_count']} output={result['output_path']}")


@datasus_app.command("normalize-sih")
def datasus_normalize_sih(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("development", "--source-manifest-hash"),
) -> None:
    from pegasus.workflows.acquire.cnes_sih import run_datasus_normalize_sih
    result = run_datasus_normalize_sih(input_path=input_path, output_path=output_path, source_manifest_hash=source_manifest_hash)
    print(f"[green]sih normalized[/green] rows={result['row_count']} output={result['output_path']}")


def dashboard_assert_read_only() -> None:
    from pegasus.dashboard.contracts import dashboard_policy_manifest

    typer.echo(json.dumps(dashboard_policy_manifest(), indent=2, sort_keys=True))


@dashboard_app.command("inspect-run")
def dashboard_inspect_run(
    run: Path = typer.Option(..., "--run"),
) -> None:
    from pegasus.workflows.report.dashboard import run_dashboard_inspect_run

    typer.echo(json.dumps(run_dashboard_inspect_run(run_dir=run), indent=2, sort_keys=True))


@dashboard_app.command("table-head")
def dashboard_table_head(
    run: Path = typer.Option(..., "--run"),
    table: str = typer.Option(..., "--table"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    from pegasus.workflows.report.dashboard import run_dashboard_table_head

    typer.echo(json.dumps(run_dashboard_table_head(run_dir=run, table_name=table, limit=limit), indent=2, sort_keys=True))

# Slice 11A acceptance hardening commands
@acceptance_app.command("plan")
def acceptance_plan() -> None:
    from pegasus.workflows.report.acceptance import run_acceptance_plan

    typer.echo(json.dumps(run_acceptance_plan(), indent=2, sort_keys=True))


@acceptance_app.command("check-run")
def acceptance_check_run(
    run: Path = typer.Option(..., "--run"),
    require_nonempty: bool = typer.Option(False, "--require-nonempty"),
) -> None:
    from pegasus.workflows.report.acceptance import run_acceptance_check_run

    result = run_acceptance_check_run(run_dir=run, require_nonempty=require_nonempty)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok", False):
        raise typer.Exit(1)



@acceptance_app.command("level3")
def acceptance_level3(
    run: Path = typer.Option(..., "--run"),
) -> None:
    from pegasus.workflows.report.acceptance import run_acceptance_level3

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
    from pegasus.workflows.acquire.source_artifacts import run_source_artifact_inspect

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
    from pegasus.workflows.acquire.source_artifacts import run_source_manifest_validate

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
    from pegasus.workflows.acquire.source_artifacts import run_source_manifest_summary

    typer.echo(json.dumps(run_source_manifest_summary(manifest=manifest), indent=2, sort_keys=True))


@source_artifacts_app.command("compile-reality-plan")
def source_artifacts_compile_reality_plan(
    source_manifest: Path | None = typer.Option(None, "--source-manifest"),
    require_materialized_external: bool = typer.Option(False, "--require-materialized-external"),
) -> None:
    from pegasus.workflows.report.compile_source import run_compile_source_reality_plan

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
    from pegasus.workflows.construct.build_substrate import run_build_substrate_from_artifacts

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
    from pegasus.workflows.construct.build_substrate import run_build_substrate_from_source_manifest

    result = run_build_substrate_from_source_manifest(source_manifest=source_manifest, output=output)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@she_app.command("substrate-summary")
def she_substrate_summary(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    from pegasus.workflows.construct.build_substrate import run_substrate_summary

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

    from pegasus.workflows.construct.efg_materialize import run_materialize_substrate_manifest

    payload = run_materialize_substrate_manifest(substrate_manifest=substrate_manifest, output=output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


@efg_app.command("attach-materialization")
def efg_attach_materialization_manifest(
    run_dir: Path = typer.Option(..., "--run-dir"),
    substrate_manifest: Path | None = typer.Option(None, "--substrate-manifest"),
) -> None:
    """Attach EFG substrate materialization metadata to an existing run bundle."""
    import json

    from pegasus.workflows.construct.efg_materialize import run_attach_efg_materialization_to_run

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

# EFG promotion CLI verbs (Slice 15D: plan/attach/apply/inspect-promotion) retired —
# legacy off-compile-path surface; promotion_plan/apply modules deleted (refactor plan §1e).

