from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.core.schemas import UserIntent
from pegasus.efg.compile_attach import attach_autonomous_efg_to_run
from pegasus.efg.dag import build_efg
from pegasus.geo.state_panel import GeoScope
from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest
from pegasus.output.bundle_manager import OutputBundleManager
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.race_bridge import RaceBridgeRegistryError, resolve_race_bridge_plan
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, load_source_artifacts_from_manifest
from pegasus.workflows.msd_inference import run_msd_inference_pipeline




def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_intent(intent_path: Path) -> tuple[dict[str, Any], UserIntent]:
    payload = json.loads(intent_path.read_text(encoding="utf-8"))
    try:
        return payload, UserIntent.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid UserIntent file {intent_path}: {exc}") from exc


def _registry_hashes() -> dict[str, str]:
    candidates = [
        Path("config/registries/registry_manifest.yaml"),
        Path("config/registries/sidra_views.yaml"),
        Path("config/registries/source_fields.yaml"),
        Path("config/registries/quality_permissions.yaml"),
        Path("config/registries/race_axis_registry.yaml"),
        Path("config/registries/race_bridge_priors.yaml"),
    ]
    return {str(path): sha256_file(path) for path in candidates if path.exists()}


def _write_compile_manifest(*, run_id: str, intent_path: Path, data_root: Path, run_dir: Path, payload: dict[str, Any]) -> Path:
    path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "compile_manifest",
                "run_id": run_id,
                "intent_path": str(intent_path),
                "run_dir": str(run_dir),
                "intent": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path




def _intent_municipality_filter_cod6(intent: UserIntent) -> str | None:
    if intent.geography.level != "municipality":
        raise ValueError("Current compile supports geography.level='municipality' only.")

    if intent.execution_scale == "smoke":
        if len(intent.geography.codes) != 1:
            raise ValueError("Municipality-scale compile requires exactly one municipality code.")
        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise ValueError(
                f"Could not resolve DATASUS cod6 for municipality {intent.geography.codes[0]!r}."
            )
        return cod6

    if intent.execution_scale == "state":
        if intent.geography.codes:
            raise ValueError(
                "State compile expects geography.codes=[] and geography.uf=[<UF>]. "
                "Explicit municipal subsets require a declared panel support contract."
            )
        if len(intent.geography.uf) != 1:
            raise ValueError("State compile requires exactly one UF in geography.uf.")
        return None

    raise ValueError(
        "Compile currently supports execution_scale='smoke' or execution_scale='state'. "
        f"Received {intent.execution_scale!r}."
    )


def _geo_scope_from_intent(intent: UserIntent, *, municipality_cod6: str | None) -> GeoScope:
    if intent.execution_scale == "state":
        if len(intent.geography.uf) != 1:
            raise ValueError("State compile requires exactly one UF in geography.uf.")
        return GeoScope.from_uf(intent.geography.uf[0], level=intent.geography.level)

    if municipality_cod6 is None:
        raise ValueError("Municipality-scale compile requires a resolved DATASUS cod6.")
    return GeoScope(
        level=intent.geography.level,
        uf=None,
        datasus_uf_prefix=str(municipality_cod6)[:2],
        ibge_uf_cod2=str(municipality_cod6)[:2],
        municipality_cod6_allowlist=frozenset({str(municipality_cod6)}),
    )


def _smoke_municipality_cod6(intent: UserIntent) -> str:
    cod6 = _intent_municipality_filter_cod6(intent)
    if cod6 is None:
        raise ValueError("Municipality helper received a state-wide intent.")
    return cod6


def _context_policy_enabled(intent: UserIntent, token: str) -> bool:
    return token in set(intent.context_policy)


def _compile_population_tensor_mode(intent: UserIntent) -> str | None:
    if intent.population_mode == "independent_population_tensor":
        return "independent_denominator"
    if intent.population_mode == "sim_informed_population_tensor":
        return "sim_informed_denominator"
    return None


def _compiler_architecture_metadata() -> dict[str, Any]:
    return {
        "schema_version": "27A.2",
        "graph_authority": "autonomous_efg_core",
        "graph_builder": "pegasus.efg.dag.build_efg",
        "numerical_materialization": "autonomous_compiler_services",
        "numerical_materializer": "pegasus.workflows.compile._run_compile_impl",
        "legacy_bootstrap_builder": None,
        "legacy_bootstrap_status": "retired_deleted",
        "legacy_graph_authority": False,
        "retired_legacy_modules": [],
        "production_runtime_authority": "autonomous_efg_core_plus_compiler_services",
    }


def _validate_compile_manifest_artifacts(
    artifacts: tuple[SourceArtifactRef, ...],
    *,
    include_cnes_sih: bool,
) -> None:
    required: set[tuple[str, str]] = {
        ("SIM-DO", "processed_events"),
        ("SINASC", "processed_events"),
        ("SIDRA", "normalized_facts"),
    }
    if include_cnes_sih:
        required.update({
            ("CNES-ST", "processed_events"),
            ("SIH-RD", "processed_events"),
        })
    available: dict[tuple[str, str], list[SourceArtifactRef]] = {}
    for artifact in artifacts:
        key = (artifact.source_system, artifact.artifact_role)
        available.setdefault(key, []).append(artifact)
        if not Path(artifact.path).is_file():
            raise FileNotFoundError(f"Production source artifact is missing: {artifact.path}")
    for system, role in sorted(required):
        matches = available.get((system, role), [])
        if len(matches) != 1:
            raise ValueError(f"Production compile requires exactly one {system}:{role} artifact; found {len(matches)}.")


def _run_compile_impl(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    intent_path = Path(intent_path)
    data_root = Path(data_root)
    from pegasus.source_artifacts.compile_policy import (
        attach_compile_source_reality,
        resolve_compile_source_reality,
    )

    compile_source_reality = resolve_compile_source_reality(
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    )
    intent_payload, intent = _load_intent(intent_path)
    municipality_cod6 = _intent_municipality_filter_cod6(intent)
    geo_scope = _geo_scope_from_intent(intent, municipality_cod6=municipality_cod6)
    if municipality_cod6 is None and str(intent.race_tensor_mode) != "decoupled":
        raise ValueError("State-level compile currently supports race_tensor_mode='decoupled' only.")
    include_cnes_sih = _context_policy_enabled(intent, "include_cnes_sih")
    population_tensor_mode = _compile_population_tensor_mode(intent)
    compiler_architecture = _compiler_architecture_metadata()
    from pegasus.workflows.stage_plan import build_compile_stage_plan

    compiler_stage_plan = build_compile_stage_plan(
        intent=intent,
        population_tensor_mode=population_tensor_mode,
        include_cnes_sih=include_cnes_sih,
        source_manifest=None if source_manifest is None else str(source_manifest),
        require_materialized_external=require_materialized_external,
    )

    try:
        race_bridge_plan = resolve_race_bridge_plan(intent=intent, municipality_cod6=municipality_cod6)
    except RaceBridgeRegistryError as exc:
        raise ValueError(f"Invalid Race Bridge registry plan for {intent_path}: {exc}") from exc
    if race_bridge_plan.status == "blocked":
        raise ValueError(f"Race Bridge mode is blocked for this compile slice: {race_bridge_plan.as_manifest()}")

    intent_hash = sha256_file(intent_path)
    run_id = f"compile_{intent_path.stem}_{utc_stamp()}_{intent_hash[:8]}"
    run_dir = Path(run_dir) if run_dir is not None else data_root / "runs" / run_id
    diagnostic_path = data_root / "diagnostics" / "compile" / f"{run_id}.telemetry.json"
    telemetry = RunTelemetry(run_id=run_id, diagnostic_path=diagnostic_path)
    bundle_manager = OutputBundleManager(run_dir=run_dir)

    source_hashes: dict[str, str] = {"intent": intent_hash}
    registry_hashes = _registry_hashes()
    if race_bridge_plan.registry_path is not None and race_bridge_plan.registry_hash is not None:
        registry_hashes[str(race_bridge_plan.registry_path)] = race_bridge_plan.registry_hash

    if compile_source_reality.compile_source_mode != "materialized_external":
        raise ValueError(
            "Production compile requires materialized_external source artifacts. "
            "Development data builders must live outside src/pegasus production workflows."
        )

    compile_manifest_path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
    if source_manifest is None:
        raise ValueError("Production compile requires a source artifact manifest.")
    autonomous_artifacts = load_source_artifacts_from_manifest(source_manifest)
    _validate_compile_manifest_artifacts(autonomous_artifacts, include_cnes_sih=include_cnes_sih)

    with telemetry.stage("datasus_manifest"):
        compile_manifest_path = _write_compile_manifest(
            run_id=run_id,
            intent_path=intent_path,
            data_root=data_root,
            run_dir=run_dir,
            payload=intent_payload,
        )
        source_hashes["compile_manifest"] = sha256_file(compile_manifest_path)

    with telemetry.stage("datasus_acquire"):
        source_hashes.update({
            f"source_artifact_{artifact.source_system}_{artifact.artifact_role}": artifact.artifact_hash or sha256_file(Path(artifact.path))
            for artifact in autonomous_artifacts
        })

    with telemetry.stage("datasus_decode"):
        source_hashes["source_manifest"] = sha256_file(Path(source_manifest))

    telemetry.set_stage("sidra_metadata", "skipped", 0.0)
    telemetry.set_stage("sidra_plan", "skipped", 0.0)
    telemetry.set_stage("sidra_fetch", "skipped", 0.0)
    telemetry.flush()

    with telemetry.stage("sidra_normalize"):
        source_hashes["sidra_facts"] = next(
            artifact.artifact_hash or sha256_file(Path(artifact.path))
            for artifact in autonomous_artifacts
            if artifact.source_system == "SIDRA" and artifact.artifact_role == "normalized_facts"
        )

    autonomous_efg_metadata: dict[str, Any] | None = None
    with telemetry.stage("she_build"):
        autonomous_substrate = build_substrate_bundle(artifacts=autonomous_artifacts)

    with telemetry.stage("efg_build"):
        autonomous_result = build_efg(
            substrate=autonomous_substrate,
            intent=intent,
            intent_constraints={"race_bridge_plan": race_bridge_plan.as_manifest()},
            operator_mode="standard",
        )
        autonomous_attach = attach_autonomous_efg_to_run(
            run_dir=run_dir,
            result=autonomous_result,
            validate=False,
            bundle=bundle_manager,
            intent=intent,
        )
        autonomous_efg_metadata = {
            **autonomous_attach.as_manifest(),
            "substrate_id": autonomous_substrate.substrate_id,
            "source_reality_mode": autonomous_substrate.source_reality_mode,
            "source_systems": sorted({artifact.source_system for artifact in autonomous_artifacts}),
            "registry_hashes": autonomous_result.registry_hashes,
            "legality_summary": autonomous_result.legality_summary,
            "precompression": autonomous_result.precompression.as_manifest(),
        }
        source_hashes["autonomous_efg_manifest"] = autonomous_attach.manifest_hash

    domain_summaries = dict(autonomous_result.domain_summaries or {})
    cnes_sih_metadata: dict[str, Any] | None = domain_summaries.get("cnes_sih")
    population_tensor_metadata: dict[str, Any] | None = domain_summaries.get("population_tensor")
    race_bridge_metadata: dict[str, Any] | None = domain_summaries.get("race_bridge")

    # MSD cutover: manual domain attachers are forbidden. CNES/SIH, maternal-child,
    # SIDRA denominators, population, and race bridge fields must be produced by
    # SHE substrate + autonomous EFG bridge/operator expansion + physical executor.
    telemetry.set_stage("race_bridge", "success" if race_bridge_metadata is not None else "skipped", 0.0)
    telemetry.flush()

    telemetry.set_stage("geo_support", "success", 0.0)
    telemetry.set_stage("q_tensor", "success", 0.0)
    if population_tensor_metadata is None:
        telemetry.set_stage("population_solver", "skipped", 0.0)
    telemetry.set_stage("stdfm", "skipped", 0.0)
    skipped_reasons = {
        **compiler_stage_plan.skip_reason_map(),
        "stdfm": "compile intent does not request latent-factor fitting",
    }
    if population_tensor_metadata is None:
        skipped_reasons["population_solver"] = "official SIDRA anchor selected by intent"

    pirs_hsic_metadata = run_msd_inference_pipeline(
        run_dir=run_dir,
        compiler_stage_plan=compiler_stage_plan,
        telemetry=telemetry,
        budget=str(intent.budget),
        bundle=bundle_manager,
    )
    telemetry.resource_summary["msd_inference_pipeline"] = {
        "manifest_path": pirs_hsic_metadata.get("manifest_path"),
        "pirs_model_status": (pirs_hsic_metadata.get("pirs_model") or {}).get("status"),
        "pirs_hsic_status": (pirs_hsic_metadata.get("pirs_hsic") or {}).get("status"),
    }
    telemetry.resource_summary["skipped_reasons"] = skipped_reasons
    telemetry.flush()

    with telemetry.stage("output_serialization"):
        (run_dir / "UserIntent.json").write_text(
            json.dumps(intent_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        run_config_path = run_dir / "RunConfig.json"
        run_config_payload = {
            "schema_version": "1.0",
            "compile_mode": "compile",
            "run_id": run_id,
            "intent_path": str(intent_path),
            "data_root": str(data_root),
            "context_policy": intent.context_policy,
            "support_policy": {
                "geography_level": intent.geography.level,
                "ibge_cod7": intent.geography.codes,
                "datasus_cod6": [municipality_cod6],
                "geo_scope": {
                    "level": geo_scope.level,
                    "uf": geo_scope.uf,
                    "datasus_uf_prefix": geo_scope.datasus_uf_prefix,
                    "ibge_uf_cod2": geo_scope.ibge_uf_cod2,
                    "expected_municipality_count": geo_scope.expected_municipality_count,
                },
                "geo_mode": intent.geo_mode,
            },
            "population_mode": intent.population_mode,
            "race_tensor_mode": intent.race_tensor_mode,
            "race_bridge_plan": race_bridge_plan.as_manifest(),
            "compiler_architecture": compiler_architecture,
            "compiler_stage_plan": compiler_stage_plan.as_manifest(),
            "registry_hashes": registry_hashes,
            "source_hashes": source_hashes,
        }
        existing_run_config = {}
        if run_config_path.exists():
            try:
                existing_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_run_config = {}
        if existing_run_config.get("maternal_child_linkage"):
            run_config_payload["maternal_child_linkage"] = existing_run_config["maternal_child_linkage"]
        if race_bridge_metadata is None and existing_run_config.get("race_bridge"):
            race_bridge_metadata = existing_run_config["race_bridge"]
        if race_bridge_metadata is not None:
            run_config_payload["race_bridge"] = race_bridge_metadata
        if cnes_sih_metadata is None and existing_run_config.get("cnes_sih"):
            cnes_sih_metadata = existing_run_config["cnes_sih"]
        if cnes_sih_metadata is not None:
            run_config_payload["cnes_sih"] = cnes_sih_metadata
        if population_tensor_metadata is None and existing_run_config.get("population_tensor"):
            population_tensor_metadata = existing_run_config["population_tensor"]
        if population_tensor_metadata is not None:
            run_config_payload["population_tensor"] = population_tensor_metadata
        if autonomous_efg_metadata is not None:
            run_config_payload["autonomous_efg"] = autonomous_efg_metadata
        run_config_path.write_text(
            json.dumps(run_config_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        manifest_extras = {
            "compile_mode": "compile",
            "compile_manifest": str(compile_manifest_path),
            "intent_path": str(intent_path),
            "maternal_child_linkage": True,
            "race_bridge_plan": race_bridge_plan.as_manifest(),
            "context_policy": intent.context_policy,
            "compiler_architecture": compiler_architecture,
            "compiler_stage_plan": compiler_stage_plan.as_manifest(),
        }
        if race_bridge_metadata is not None:
            manifest_extras["race_bridge"] = race_bridge_metadata
        if cnes_sih_metadata is not None:
            manifest_extras["cnes_sih"] = cnes_sih_metadata
        if population_tensor_metadata is not None:
            manifest_extras["population_tensor"] = population_tensor_metadata
        if autonomous_efg_metadata is not None:
            manifest_extras["autonomous_efg"] = autonomous_efg_metadata
        write_reproducibility_manifest(
            run_dir=run_dir,
            run_id=run_id,
            intent_hash=intent_hash,
            source_hashes=source_hashes,
            registry_hashes=registry_hashes,
            telemetry=telemetry,
            extras=manifest_extras,
        )

    final_extras = {
        "compile_mode": "compile",
        "compile_manifest": str(compile_manifest_path),
        "intent_path": str(intent_path),
        "maternal_child_linkage": True,
        "race_bridge_plan": race_bridge_plan.as_manifest(),
        "context_policy": intent.context_policy,
        "compiler_architecture": compiler_architecture,
        "compiler_stage_plan": compiler_stage_plan.as_manifest(),
    }
    if race_bridge_metadata is not None:
        final_extras["race_bridge"] = race_bridge_metadata
    if cnes_sih_metadata is not None:
        final_extras["cnes_sih"] = cnes_sih_metadata
    if population_tensor_metadata is not None:
        final_extras["population_tensor"] = population_tensor_metadata
    if autonomous_efg_metadata is not None:
        final_extras["autonomous_efg"] = autonomous_efg_metadata
    if "pirs_hsic_metadata" in locals() and pirs_hsic_metadata is not None:
        final_extras["msd_inference_pipeline"] = pirs_hsic_metadata
    write_reproducibility_manifest(
        run_dir=run_dir,
        run_id=run_id,
        intent_hash=intent_hash,
        source_hashes=source_hashes,
        registry_hashes=registry_hashes,
        telemetry=telemetry,
        extras=final_extras,
    )

    # Phase E boundary: first-class tables are valid only after atomic bundle flush.
    with telemetry.stage("output_bundle_flush"):
        attach_compile_source_reality(run_dir=run_dir, source_reality=compile_source_reality)
        bundle_manager.collect_missing_from_run(run_dir)
        run_dir = bundle_manager.flush_to_disk(run_dir)
    validation = validate_output_bundle(run_dir=str(run_dir))
    return {
        "status": "success" if validation.ok else "failed",
        "run_id": run_id,
        "run_dir": run_dir,
        "intent": intent,
        "validation": validation,
        "source_hashes": source_hashes,
        "registry_hashes": registry_hashes,
        "telemetry": telemetry.model(),
        "race_bridge_plan": race_bridge_plan.as_manifest(),
        "race_bridge": race_bridge_metadata,
        "cnes_sih": cnes_sih_metadata,
        "population_tensor": population_tensor_metadata,
        "autonomous_efg": autonomous_efg_metadata,
        "compiler_architecture": compiler_architecture,
        "compiler_stage_plan": compiler_stage_plan.as_manifest(),
    }

# ---- Slice 13C compile/substrate contract consolidation ----
def run_compile(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    # Single public compile boundary. The actual smoke compiler remains in
    # _run_compile_impl(); this wrapper only attaches SHE substrate metadata
    # when source artifact manifests are supplied.
    result = _run_compile_impl(
        intent_path=intent_path,
        run_dir=run_dir,
        data_root=data_root,
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    )
    if not isinstance(result, dict):
        return result

    run_path = result.get("run_dir")
    if run_path is None:
        result["substrate_gate"] = {
            "status": "failed_nonfatal",
            "reason": "compile_result_missing_run_dir",
        }
        return result

    from pegasus.output.validate import validate_output_bundle as _validate_output_bundle
    from pegasus.workflows.build_substrate import run_attach_substrate_to_run as _run_attach_substrate_to_run

    substrate_summary = _run_attach_substrate_to_run(run_dir=run_path, source_manifest=source_manifest)
    result["substrate_gate"] = substrate_summary
    result["validation"] = _validate_output_bundle(run_dir=str(run_path))
    return result
