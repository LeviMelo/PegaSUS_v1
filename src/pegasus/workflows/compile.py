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
from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
from pegasus.output.cnes_sih_compile_attach import attach_cnes_sih_compile_fields
from pegasus.output.maternal_child_compile_attach import attach_maternal_child_compile_fields
from pegasus.output.population_tensor_compile_attach import attach_population_tensor_compile_fields
from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.race_bridge import RaceBridgeRegistryError, resolve_race_bridge_plan
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.she.substrate import build_substrate_bundle
from pegasus.workflows.cnes_sih import run_datasus_normalize_cnes, run_datasus_normalize_sih
from pegasus.workflows.datasus import run_datasus_normalize_sim
from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture
from pegasus.workflows.race_bridge import run_attach_race_bridge
from pegasus.workflows.sinasc import run_datasus_normalize_sinasc


SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD: list[dict[str, str]] = [
    {
        "NC": "Nível Territorial (Código)",
        "NN": "Nível Territorial",
        "MC": "Unidade de Medida (Código)",
        "MN": "Unidade de Medida",
        "V": "Valor",
        "D1C": "Município (Código)",
        "D1N": "Município",
        "D2C": "Ano (Código)",
        "D2N": "Ano",
        "D3C": "Variável (Código)",
        "D3N": "Variável",
        "D4C": "Sexo (Código)",
        "D4N": "Sexo",
        "D5C": "Cor ou raça (Código)",
        "D5N": "Cor ou raça",
        "D6C": "Idade (Código)",
        "D6N": "Idade",
    },
    {
        "NC": "6",
        "NN": "Município",
        "MC": "45",
        "MN": "Pessoas",
        "V": "957916",
        "D1C": "2704302",
        "D1N": "Maceió (AL)",
        "D2C": "2022",
        "D2N": "2022",
        "D3C": "93",
        "D3N": "População residente",
        "D4C": "6794",
        "D4N": "Total",
        "D5C": "95251",
        "D5N": "Total",
        "D6C": "100362",
        "D6N": "Total",
    },
]


SIDRA_POPULATION_MACEIO_CHUNK_REQUEST: dict[str, Any] = {
    "table_id": "9606",
    "variables": ["93"],
    "periods": ["2022"],
    "locality_level": "N6",
    "localities": ["2704302"],
    "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
}


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
    facts = normalize_sidra_payload_to_facts(
        SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD,
        table_id="9606",
        request_hash=content_hash(SIDRA_POPULATION_MACEIO_CHUNK_REQUEST),
        metadata_hash=content_hash({"metadata": "compile_smoke_sidra_9606_maceio_total_v1"}),
        chunk_request=SIDRA_POPULATION_MACEIO_CHUNK_REQUEST,
        unit_by_variable=None,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    write_facts_parquet(facts, output_path=output_path)
    return output_path


def _smoke_municipality_cod6(intent: UserIntent) -> str:
    if intent.execution_scale != "smoke":
        raise ValueError("Compile smoke supports execution_scale='smoke' only.")
    if intent.geography.level != "municipality":
        raise ValueError("Compile smoke requires geography.level='municipality'.")
    if len(intent.geography.codes) != 1:
        raise ValueError("Compile smoke requires exactly one municipality code.")
    cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
    if cod6 is None:
        raise ValueError(f"Could not convert intent municipality code to DATASUS cod6: {intent.geography.codes[0]!r}")
    return cod6


def _context_policy_enabled(intent: UserIntent, token: str) -> bool:
    return token in set(intent.context_policy)


def _compile_population_tensor_mode(intent: UserIntent) -> str | None:
    if intent.population_mode == "independent_population_tensor":
        return "independent_denominator"
    if intent.population_mode == "sim_informed_population_tensor":
        return "sim_informed_denominator"
    return None

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
    municipality_cod6 = _smoke_municipality_cod6(intent)
    include_cnes_sih = _context_policy_enabled(intent, "include_cnes_sih")
    population_tensor_mode = _compile_population_tensor_mode(intent)

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
    sidra_facts_path = data_root / "processed" / "sidra" / "facts" / "9606" / "compile_smoke_maceio.parquet"

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
        run_datasus_normalize_sim(
            input_path=raw_cache_path,
            output_path=sim_events_path,
            source_manifest_hash=source_hashes["compile_manifest"],
        )
        run_datasus_normalize_sinasc(
            input_path=raw_sinasc_cache_path,
            output_path=sinasc_events_path,
            source_manifest_hash=source_hashes["compile_manifest"],
        )
        source_hashes["sim_processed_events"] = sha256_file(sim_events_path)
        source_hashes["sinasc_processed_events"] = sha256_file(sinasc_events_path)
        if include_cnes_sih:
            run_datasus_normalize_cnes(
                input_path=raw_cnes_cache_path,
                output_path=cnes_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            run_datasus_normalize_sih(
                input_path=raw_sih_cache_path,
                output_path=sih_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            source_hashes["cnes_processed_events"] = sha256_file(cnes_events_path)
            source_hashes["sih_processed_events"] = sha256_file(sih_events_path)

    telemetry.set_stage("sidra_metadata", "skipped", 0.0)
    telemetry.set_stage("sidra_plan", "skipped", 0.0)
    telemetry.set_stage("sidra_fetch", "skipped", 0.0)
    telemetry.flush()

    with telemetry.stage("sidra_normalize"):
        _write_sidra_smoke_facts(output_path=sidra_facts_path)
        source_hashes["sidra_facts"] = sha256_file(sidra_facts_path)

    autonomous_efg_metadata: dict[str, Any] | None = None
    with telemetry.stage("efg_build"):
        run_build_sim_fixture(
            sim_events_path=sim_events_path,
            run_dir=run_dir,
            municipality_cod6=municipality_cod6,
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
    with telemetry.stage("she_build"):
        run_attach_sidra_denominator(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts_path,
        )
        attach_maternal_child_compile_fields(
            run_dir=run_dir,
            sinasc_events_path=sinasc_events_path,
            sim_events_path=sim_events_path,
            municipality_cod6=municipality_cod6,
        )
        source_hashes["maternal_child_linkage_summary"] = sha256_file(
            Path(run_dir) / "Tables" / "maternal_child_linkage_summary.parquet"
        )
        if include_cnes_sih:
            cnes_sih_result = attach_cnes_sih_compile_fields(
                run_dir=run_dir,
                cnes_events_path=cnes_events_path,
                sih_events_path=sih_events_path,
                municipality_cod6=municipality_cod6,
            )
            cnes_sih_metadata = cnes_sih_result["cnes_sih"]
            source_hashes.update(cnes_sih_metadata.get("source_hashes", {}))

    if population_tensor_mode is not None:
        with telemetry.stage("population_solver"):
            population_tensor_metadata = attach_population_tensor_compile_fields(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts_path,
                mode=population_tensor_mode,
            )
            source_hashes.update(
                {
                    f"population_tensor_{key}": value
                    for key, value in population_tensor_metadata.get("source_hashes", {}).items()
                }
            )
            diagnostics_path = Path(run_dir) / "Tables" / "population_tensor_diagnostics.parquet"
            if diagnostics_path.exists():
                source_hashes["population_tensor_diagnostics"] = sha256_file(diagnostics_path)

    race_bridge_metadata: dict[str, Any] | None = None
    if race_bridge_plan.requires_attach:
        assert race_bridge_plan.prior_path is not None
        with telemetry.stage("race_bridge"):
            bridge_result = run_attach_race_bridge(
                run_dir=run_dir,
                sim_events_path=sim_events_path,
                bridge_prior_path=race_bridge_plan.prior_path,
                municipality_cod6=municipality_cod6,
            )
            if not bridge_result["validation"].ok:
                raise RuntimeError("Race bridge attachment invalidated run bundle: " + "; ".join(bridge_result["validation"].errors))
            race_bridge_metadata = bridge_result.get("race_bridge")
            source_hashes["race_bridge_prior"] = sha256_file(race_bridge_plan.prior_path)
            summary_path = Path(run_dir) / "Tables" / "race_bridge_summary.parquet"
            if summary_path.exists():
                source_hashes["race_bridge_summary"] = sha256_file(summary_path)
    else:
        telemetry.set_stage("race_bridge", "skipped", 0.0)
        telemetry.flush()

    telemetry.set_stage("geo_support", "success", 0.0)
    telemetry.set_stage("q_tensor", "success", 0.0)
    if population_tensor_metadata is None:
        telemetry.block("population_solver", reason="official SIDRA anchor smoke path; tensor solver scaffold remains blocked")
    telemetry.block("stdfm", reason="ST-DFM scaffold remains blocked for compile smoke")
    telemetry.block("pirs_model", reason="PIRS model stage is not invoked in compile smoke")
    telemetry.block("pirs_hsic", reason="PIRS HSIC stage is not invoked in compile smoke")
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
                "geo_mode": intent.geo_mode,
            },
            "population_mode": intent.population_mode,
            "race_tensor_mode": intent.race_tensor_mode,
            "race_bridge_plan": race_bridge_plan.as_manifest(),
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
    }
    if race_bridge_metadata is not None:
        final_extras["race_bridge"] = race_bridge_metadata
    if cnes_sih_metadata is not None:
        final_extras["cnes_sih"] = cnes_sih_metadata
    if population_tensor_metadata is not None:
        final_extras["population_tensor"] = population_tensor_metadata
    if autonomous_efg_metadata is not None:
        final_extras["autonomous_efg"] = autonomous_efg_metadata
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
