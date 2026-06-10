from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.core.schemas import UserIntent
from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest
from pegasus.output.validate import validate_output_bundle
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.workflows.datasus import run_datasus_normalize_sim
from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture


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
    "classifications": {
        "86": ["95251"],
        "2": ["6794"],
        "287": ["100362"],
    },
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
        raise ValueError("Slice 2D compile supports execution_scale='smoke' only.")
    if intent.geography.level != "municipality":
        raise ValueError("Slice 2D compile smoke requires geography.level='municipality'.")
    if len(intent.geography.codes) != 1:
        raise ValueError("Slice 2D compile smoke requires exactly one municipality code.")
    cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
    if cod6 is None:
        raise ValueError(f"Could not convert intent municipality code to DATASUS cod6: {intent.geography.codes[0]!r}")
    return cod6


def run_compile(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
) -> dict[str, Any]:
    intent_path = Path(intent_path)
    data_root = Path(data_root)
    intent_payload, intent = _load_intent(intent_path)
    municipality_cod6 = _smoke_municipality_cod6(intent)

    intent_hash = sha256_file(intent_path)
    run_id = f"compile_{intent_path.stem}_{utc_stamp()}_{intent_hash[:8]}"
    run_dir = Path(run_dir) if run_dir is not None else data_root / "runs" / run_id
    diagnostic_path = data_root / "diagnostics" / "compile" / f"{run_id}.telemetry.json"
    telemetry = RunTelemetry(run_id=run_id, diagnostic_path=diagnostic_path)

    source_hashes: dict[str, str] = {"intent": intent_hash}
    registry_hashes = _registry_hashes()

    raw_fixture_source = Path("tests/fixtures/datasus/sim_do_fixture.csv")
    if not raw_fixture_source.exists():
        raise FileNotFoundError(f"Missing SIM smoke fixture: {raw_fixture_source}")

    compile_manifest_path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
    raw_cache_path = data_root / "raw" / "datasus" / "SIM-DO" / "fixture" / "sim_do_fixture.csv"
    sim_events_path = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
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
        shutil.copyfile(raw_fixture_source, raw_cache_path)
        source_hashes["sim_raw_fixture"] = sha256_file(raw_cache_path)

    with telemetry.stage("datasus_decode"):
        run_datasus_normalize_sim(
            input_path=raw_cache_path,
            output_path=sim_events_path,
            source_manifest_hash=source_hashes["compile_manifest"],
        )
        source_hashes["sim_processed_events"] = sha256_file(sim_events_path)

    telemetry.set_stage("sidra_metadata", "skipped", 0.0)
    telemetry.set_stage("sidra_plan", "skipped", 0.0)
    telemetry.set_stage("sidra_fetch", "skipped", 0.0)
    telemetry.flush()

    with telemetry.stage("sidra_normalize"):
        _write_sidra_smoke_facts(output_path=sidra_facts_path)
        source_hashes["sidra_facts"] = sha256_file(sidra_facts_path)

    with telemetry.stage("efg_build"):
        run_build_sim_fixture(
            sim_events_path=sim_events_path,
            run_dir=run_dir,
            municipality_cod6=municipality_cod6,
        )

    with telemetry.stage("she_build"):
        run_attach_sidra_denominator(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts_path,
        )

    telemetry.set_stage("geo_support", "success", 0.0)
    telemetry.set_stage("q_tensor", "success", 0.0)
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
        (run_dir / "RunConfig.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "compile_mode": "smoke",
                    "run_id": run_id,
                    "intent_path": str(intent_path),
                    "data_root": str(data_root),
                    "support_policy": {
                        "geography_level": intent.geography.level,
                        "ibge_cod7": intent.geography.codes,
                        "datasus_cod6": [municipality_cod6],
                        "geo_mode": intent.geo_mode,
                    },
                    "population_mode": intent.population_mode,
                    "race_tensor_mode": intent.race_tensor_mode,
                    "registry_hashes": registry_hashes,
                    "source_hashes": source_hashes,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_reproducibility_manifest(
            run_dir=run_dir,
            run_id=run_id,
            intent_hash=intent_hash,
            source_hashes=source_hashes,
            registry_hashes=registry_hashes,
            telemetry=telemetry,
            extras={
                "compile_mode": "smoke",
                "compile_manifest": str(compile_manifest_path),
                "intent_path": str(intent_path),
            },
        )

    with telemetry.stage("output_validation"):
        result = validate_output_bundle(run_dir=str(run_dir))
        if not result.ok:
            raise RuntimeError("Compile smoke produced invalid output bundle: " + "; ".join(result.errors))

    write_reproducibility_manifest(
        run_dir=run_dir,
        run_id=run_id,
        intent_hash=intent_hash,
        source_hashes=source_hashes,
        registry_hashes=registry_hashes,
        telemetry=telemetry,
        extras={
            "compile_mode": "smoke",
            "compile_manifest": str(compile_manifest_path),
            "intent_path": str(intent_path),
        },
    )

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
    }
