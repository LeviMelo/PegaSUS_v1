from __future__ import annotations

import json
import shutil
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
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.she.substrate import build_substrate_bundle
from pegasus.workflows.cnes_sih import run_datasus_normalize_cnes, run_datasus_normalize_sih
from pegasus.workflows.datasus import run_datasus_normalize_sim
from pegasus.workflows.build_efg import build_sim_compiler_run
from pegasus.workflows.efg import run_attach_sidra_denominator
from pegasus.workflows.sinasc import run_datasus_normalize_sinasc
from pegasus.workflows.msd_inference import run_msd_inference_pipeline


SIDRA_COMPILE_SMOKE_FIXTURE = Path("tests/fixtures/sidra/compile_smoke_sidra_9606_population.json")


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
                "kind": "compile_smoke_manifest",
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


def _write_sidra_smoke_facts(*, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(SIDRA_COMPILE_SMOKE_FIXTURE.read_text(encoding="utf-8"))
    payload = fixture["payload"]
    chunk_request = fixture["chunk_request"]
    facts = normalize_sidra_payload_to_facts(
        payload,
        table_id="9606",
        request_hash=content_hash(chunk_request),
        metadata_hash=content_hash(fixture.get("metadata", {})),
        chunk_request=chunk_request,
        unit_by_variable=None,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    write_facts_parquet(facts, output_path=output_path)
    return output_path


def _intent_municipality_filter_cod6(intent: UserIntent) -> str | None:
    """Resolve the optional DATASUS municipality filter for compile inputs.

    Smoke runs remain single-municipality runs. State runs over a UF deliberately
    return None so SIM/SINASC are not filtered down to one municipality.
    This is a state-level aggregate support, not yet a per-municipality panel.
    """
    if intent.geography.level != "municipality":
        raise ValueError("Current compile supports geography.level='municipality' only.")

    if intent.execution_scale == "smoke":
        if len(intent.geography.codes) != 1:
            raise ValueError("Compile smoke requires exactly one municipality code.")
        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise ValueError(f"Could not resolve municipality DATASUS cod6 for {intent.geography.codes[0]!r}.")
        return cod6

    if intent.execution_scale == "state":
        if intent.geography.codes:
            raise ValueError(
                "State compile currently expects geography.codes=[] and geography.uf=[<UF>]. "
                "Explicit multi-code municipal subsets require panel support and are not implemented in this slice."
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
        raise ValueError("Smoke compile requires a resolved DATASUS municipality code.")
    return GeoScope(
        level=intent.geography.level,
        uf=None,
        datasus_uf_prefix=str(municipality_cod6)[:2],
        ibge_uf_cod2=str(municipality_cod6)[:2],
        municipality_cod6_allowlist=frozenset({str(municipality_cod6)}),
    )


def _smoke_municipality_cod6(intent: UserIntent) -> str:
    """Legacy strict helper retained for tests and smoke-only callers."""
    cod6 = _intent_municipality_filter_cod6(intent)
    if cod6 is None:
        raise ValueError("Compile smoke helper received a non-smoke/state-wide intent.")
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
        "schema_version": "26A.1",
        "graph_authority": "autonomous_efg_core",
        "graph_builder": "pegasus.efg.dag.build_efg",
        "numerical_materialization": "autonomous_compiler_services",
        "numerical_materializer": "pegasus.workflows.compile._run_compile_impl",
        "legacy_bootstrap_builder": None,
        "legacy_bootstrap_status": "quarantined_fixture_only",
        "legacy_graph_authority": False,
        "fixture_compatibility_modules": [
            "pegasus.output.sim_efg_bundle",
            "pegasus.output.sinasc_efg_bundle",
            "pegasus.output.cnes_sih_efg_bundle",
            "pegasus.output.population_tensor_bundle",
            "pegasus.output.sidra_stdfm_bundle",
            "pegasus.output.pirs_bundle",
            "pegasus.output.hsic_bundle",
        ],
        "production_runtime_authority": "autonomous_efg_core_plus_compiler_services",
    }


def _external_compile_inputs(source_manifest: str | Path, *, include_cnes_sih: bool) -> dict[str, Path]:
    payload = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Source artifact manifest must contain an artifacts list.")
    required = {
        "sim_events": ("SIM-DO", "processed_events"),
        "sinasc_events": ("SINASC", "processed_events"),
        "sidra_facts": ("SIDRA", "normalized_facts"),
    }
    if include_cnes_sih:
        required.update({
            "cnes_events": ("CNES-ST", "processed_events"),
            "sih_events": ("SIH-RD", "processed_events"),
        })
    resolved: dict[str, Path] = {}
    for key, (system, role) in required.items():
        matches = [Path(str(item.get("path"))) for item in artifacts if item.get("source_system") == system and item.get("artifact_role") == role]
        if len(matches) != 1:
            raise ValueError(f"Production compile requires exactly one {system}:{role} artifact; found {len(matches)}.")
        if not matches[0].is_file():
            raise FileNotFoundError(f"Production source artifact is missing: {matches[0]}")
        resolved[key] = matches[0]
    return resolved


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

    source_hashes: dict[str, str] = {"intent": intent_hash}
    registry_hashes = _registry_hashes()
    if race_bridge_plan.registry_path is not None and race_bridge_plan.registry_hash is not None:
        registry_hashes[str(race_bridge_plan.registry_path)] = race_bridge_plan.registry_hash

    raw_fixture_source = Path("tests/fixtures/datasus/sim_do_fixture.csv")
    raw_sinasc_fixture_source = Path("tests/fixtures/datasus/sinasc_fixture.csv")
    raw_cnes_fixture_source = Path("tests/fixtures/datasus/cnes_st_fixture.csv")
    raw_sih_fixture_source = Path("tests/fixtures/datasus/sih_rd_fixture.csv")
    if compile_source_reality.compile_source_mode != "materialized_external":
        if not raw_fixture_source.exists():
            raise FileNotFoundError(f"Missing SIM smoke fixture: {raw_fixture_source}")
        if not raw_sinasc_fixture_source.exists():
            raise FileNotFoundError(f"Missing SINASC smoke fixture: {raw_sinasc_fixture_source}")
        if include_cnes_sih and not raw_cnes_fixture_source.exists():
            raise FileNotFoundError(f"Missing CNES-ST smoke fixture: {raw_cnes_fixture_source}")
        if include_cnes_sih and not raw_sih_fixture_source.exists():
            raise FileNotFoundError(f"Missing SIH-RD smoke fixture: {raw_sih_fixture_source}")

    compile_manifest_path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
    raw_cache_path = data_root / "raw" / "datasus" / "SIM-DO" / "fixture" / "sim_do_fixture.csv"
    raw_sinasc_cache_path = data_root / "raw" / "datasus" / "SINASC" / "fixture" / "sinasc_fixture.csv"
    raw_cnes_cache_path = data_root / "raw" / "datasus" / "CNES-ST" / "fixture" / "cnes_st_fixture.csv"
    raw_sih_cache_path = data_root / "raw" / "datasus" / "SIH-RD" / "fixture" / "sih_rd_fixture.csv"
    sim_events_path = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
    sinasc_events_path = data_root / "processed" / "datasus" / "SINASC" / "fixture" / "sinasc_events.parquet"
    cnes_events_path = data_root / "processed" / "datasus" / "CNES-ST" / "fixture" / "cnes_events.parquet"
    sih_events_path = data_root / "processed" / "datasus" / "SIH-RD" / "fixture" / "sih_events.parquet"
    sidra_facts_path = data_root / "processed" / "sidra" / "facts" / "9606" / "compile_smoke_population.parquet"
    external_inputs: dict[str, Path] = {}
    if compile_source_reality.compile_source_mode == "materialized_external":
        if source_manifest is None:
            raise ValueError("materialized_external compile requires a source manifest path")
        external_inputs = _external_compile_inputs(source_manifest, include_cnes_sih=include_cnes_sih)
        sim_events_path = external_inputs["sim_events"]
        sinasc_events_path = external_inputs["sinasc_events"]
        sidra_facts_path = external_inputs["sidra_facts"]
        if include_cnes_sih:
            cnes_events_path = external_inputs["cnes_events"]
            sih_events_path = external_inputs["sih_events"]

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
        if external_inputs:
            source_hashes.update({f"external_{key}": sha256_file(path) for key, path in external_inputs.items()})
        else:
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            raw_sinasc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_fixture_source, raw_cache_path)
            shutil.copyfile(raw_sinasc_fixture_source, raw_sinasc_cache_path)
            source_hashes["sim_raw_fixture"] = sha256_file(raw_cache_path)
            source_hashes["sinasc_raw_fixture"] = sha256_file(raw_sinasc_cache_path)
            if include_cnes_sih:
                raw_cnes_cache_path.parent.mkdir(parents=True, exist_ok=True)
                raw_sih_cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(raw_cnes_fixture_source, raw_cnes_cache_path)
                shutil.copyfile(raw_sih_fixture_source, raw_sih_cache_path)
                source_hashes["cnes_raw_fixture"] = sha256_file(raw_cnes_cache_path)
                source_hashes["sih_raw_fixture"] = sha256_file(raw_sih_cache_path)

    with telemetry.stage("datasus_decode"):
        if not external_inputs:
            run_datasus_normalize_sim(input_path=raw_cache_path, output_path=sim_events_path, source_manifest_hash=source_hashes["compile_manifest"])
            run_datasus_normalize_sinasc(input_path=raw_sinasc_cache_path, output_path=sinasc_events_path, source_manifest_hash=source_hashes["compile_manifest"])
        source_hashes["sim_processed_events"] = sha256_file(sim_events_path)
        source_hashes["sinasc_processed_events"] = sha256_file(sinasc_events_path)
        if include_cnes_sih:
            if not external_inputs:
                run_datasus_normalize_cnes(input_path=raw_cnes_cache_path, output_path=cnes_events_path, source_manifest_hash=source_hashes["compile_manifest"])
                run_datasus_normalize_sih(input_path=raw_sih_cache_path, output_path=sih_events_path, source_manifest_hash=source_hashes["compile_manifest"])
            source_hashes["cnes_processed_events"] = sha256_file(cnes_events_path)
            source_hashes["sih_processed_events"] = sha256_file(sih_events_path)

    telemetry.set_stage("sidra_metadata", "skipped", 0.0)
    telemetry.set_stage("sidra_plan", "skipped", 0.0)
    telemetry.set_stage("sidra_fetch", "skipped", 0.0)
    telemetry.flush()

    with telemetry.stage("sidra_normalize"):
        if not external_inputs:
            _write_sidra_smoke_facts(output_path=sidra_facts_path)
        source_hashes["sidra_facts"] = sha256_file(sidra_facts_path)

    autonomous_efg_metadata: dict[str, Any] | None = None
    with telemetry.stage("efg_build"):
        build_sim_compiler_run(
            sim_events_path=sim_events_path,
            run_dir=run_dir,
            municipality_cod6=municipality_cod6,
            datasus_uf_prefix=geo_scope.datasus_uf_prefix,
            source_mode=compile_source_reality.compile_source_mode,
        )
        provenance_mode = (
            "fixture"
            if compile_source_reality.compile_source_mode == "fixture_only"
            else compile_source_reality.compile_source_mode
        )
        autonomous_artifacts: list[dict[str, Any]] = [
            {
                "path": str(sim_events_path),
                "source_system": "SIM-DO",
                "artifact_role": "processed_events",
                "provenance_mode": provenance_mode,
                "source_manifest_hash": source_hashes["compile_manifest"],
                "artifact_hash": source_hashes["sim_processed_events"],
            },
            {
                "path": str(sinasc_events_path),
                "source_system": "SINASC",
                "artifact_role": "processed_events",
                "provenance_mode": provenance_mode,
                "source_manifest_hash": source_hashes["compile_manifest"],
                "artifact_hash": source_hashes["sinasc_processed_events"],
            },
            {
                "path": str(sidra_facts_path),
                "source_system": "SIDRA",
                "artifact_role": "normalized_facts",
                "provenance_mode": provenance_mode,
                "source_manifest_hash": source_hashes["compile_manifest"],
                "artifact_hash": source_hashes["sidra_facts"],
            },
        ]
        if include_cnes_sih:
            autonomous_artifacts.extend([
                {
                    "path": str(cnes_events_path),
                    "source_system": "CNES-ST",
                    "artifact_role": "processed_facility_periods",
                    "provenance_mode": provenance_mode,
                    "source_manifest_hash": source_hashes["compile_manifest"],
                    "artifact_hash": source_hashes["cnes_processed_events"],
                },
                {
                    "path": str(sih_events_path),
                    "source_system": "SIH-RD",
                    "artifact_role": "processed_admissions",
                    "provenance_mode": provenance_mode,
                    "source_manifest_hash": source_hashes["compile_manifest"],
                    "artifact_hash": source_hashes["sih_processed_events"],
                },
            ])
        autonomous_substrate = build_substrate_bundle(artifacts=autonomous_artifacts)
        autonomous_result = build_efg(
            substrate=autonomous_substrate,
            intent=intent,
            operator_mode="standard",
        )
        autonomous_attach = attach_autonomous_efg_to_run(
            run_dir=run_dir,
            result=autonomous_result,
            validate=False,
            bundle=bundle_manager,
        )
        autonomous_efg_metadata = {
            **autonomous_attach.as_manifest(),
            "substrate_id": autonomous_substrate.substrate_id,
            "source_reality_mode": autonomous_substrate.source_reality_mode,
            "source_systems": sorted({artifact["source_system"] for artifact in autonomous_artifacts}),
            "registry_hashes": autonomous_result.registry_hashes,
            "legality_summary": autonomous_result.legality_summary,
            "precompression": autonomous_result.precompression.as_manifest(),
        }
        source_hashes["autonomous_efg_manifest"] = autonomous_attach.manifest_hash

    cnes_sih_metadata: dict[str, Any] | None = None
    population_tensor_metadata: dict[str, Any] | None = None
    race_bridge_metadata: dict[str, Any] | None = None

    # MSD cutover: manual domain attachers are forbidden. CNES/SIH, maternal-child,
    # SIDRA denominators, population, and race bridge fields must be produced by
    # SHE substrate + autonomous EFG bridge/operator expansion + physical executor.
    telemetry.set_stage("she_build", "success", 0.0)
    telemetry.set_stage("race_bridge", "blocked" if race_bridge_plan.requires_attach else "skipped", 0.0)
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
            "compile_mode": "smoke",
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
            "compile_mode": "smoke",
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

    with telemetry.stage("output_validation"):
        result = validate_output_bundle(run_dir=str(run_dir))
        if not result.ok:
            raise RuntimeError("Compile smoke produced invalid output bundle: " + "; ".join(result.errors))

    final_extras = {
        "compile_mode": "smoke",
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

    if source_manifest is None:
        result.setdefault(
            "substrate_gate",
            {
                "status": "not_evaluated",
                "reason": "no_source_manifest_supplied_to_compile",
                "source_reality_mode": "no_manifest",
                "registry_backed": None,
            },
        )
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
