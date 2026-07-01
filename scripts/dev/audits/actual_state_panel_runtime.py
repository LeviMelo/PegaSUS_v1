from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash
from pegasus.dashboard.read_only import bundle_overview, read_table_head
from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.datasus.subprocess import DatasusConfig
from pegasus.output.validate import validate_output_bundle
from pegasus.sidra.api import SidraClient
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest
from pegasus.workflows.cnes_sih import run_datasus_normalize_cnes, run_datasus_normalize_sih
from pegasus.workflows.compile import run_compile
from pegasus.workflows.datasus import run_datasus_normalize_sim
from pegasus.workflows.sinasc import run_datasus_normalize_sinasc

ROOT = Path(__file__).resolve().parents[3]
R_LIBRARY = ROOT / ".r-library"

DATASUS_SYSTEMS = ("SIM-DO", "SINASC", "CNES-ST", "SIH-RD")
EXPECTED_SOURCE_SYSTEMS = ["CNES-ST", "SIDRA", "SIH-RD", "SIM-DO", "SINASC"]

MANDATORY_FIELD_NAMES = {
    "SIMDeathsAll",
    "SIMCrudeMortalitySIDRAOfficial",
    "SIDRAPopulationTotalAnchor",
    "SINASCLiveBirthsAll",
    "SINASCCrudeBirthRateSIDRAOfficial",
    "SINASCLowBirthWeightPrevalence",
    "SINASCPrematurityPrevalence",
    "SINASCCongenitalAnomalyPrevalence",
    "SIMInfantMortalitySINASCBirths",
    "SIMNeonatalMortalitySINASCBirths",
    "SIMPostNeonatalMortalitySINASCBirths",
    "CNESFacilitiesAll",
    "SIHHospitalAdmissionsAll",
    "SIHInpatientDeaths",
    "SIHInpatientFatality",
}

NORMALIZED_NAMES = {
    "SIM-DO": "sim_events.parquet",
    "SINASC": "sinasc_events.parquet",
    "CNES-ST": "cnes_events.parquet",
    "SIH-RD": "sih_events.parquet",
}


def _find_rscript() -> str | None:
    discovered = shutil.which("Rscript")
    if discovered:
        return discovered
    if sys.platform == "win32":
        roots = [Path("C:/Program Files/R"), Path("C:/Program Files (x86)/R")]
        candidates = [path for root in roots if root.exists() for path in root.glob("R-*/bin/Rscript.exe")]
        if candidates:
            return str(sorted(candidates)[-1])
    return None


def _r_probe(rscript: str | None) -> dict[str, Any]:
    result = {
        "rscript_found": bool(rscript),
        "rscript_path": rscript,
        "microdatasus_available": False,
        "read_dbc_available": False,
        "dplyr_available": False,
        "arrow_available": False,
        "jsonlite_available": False,
        "r_probe_error": None,
    }
    if not rscript:
        return result
    library_literal = json.dumps(str(R_LIBRARY).replace("\\", "/"))
    expression = (
        f".libPaths(c({library_literal},.libPaths()));"
        "cat(requireNamespace('microdatasus',quietly=TRUE),"
        "requireNamespace('read.dbc',quietly=TRUE),"
        "requireNamespace('dplyr',quietly=TRUE),"
        "requireNamespace('arrow',quietly=TRUE),"
        "requireNamespace('jsonlite',quietly=TRUE),sep='|')"
    )
    process = subprocess.run([rscript, "--vanilla", "-e", expression], capture_output=True, text=True, check=False, timeout=90)
    values = process.stdout.strip().split("|")
    if len(values) == 5:
        result.update(
            microdatasus_available=values[0] == "TRUE",
            read_dbc_available=values[1] == "TRUE",
            dplyr_available=values[2] == "TRUE",
            arrow_available=values[3] == "TRUE",
            jsonlite_available=values[4] == "TRUE",
        )
    if process.returncode != 0:
        result["r_probe_error"] = process.stderr.strip()[:2000] or f"R package probe exited {process.returncode}"
    elif not all(result[key] for key in ("microdatasus_available", "read_dbc_available", "dplyr_available", "arrow_available", "jsonlite_available")):
        result["r_probe_error"] = "Required R packages are absent from the vanilla R library path."
    return result


def _parse_sidra_number(value: Any) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("blank SIDRA numeric value")
    normalized = text.replace(".", "").replace(",", ".") if "," in text else text
    return float(normalized)


def _sidra_facts(*, root: Path, localities: list[str]) -> tuple[Path, bool, int]:
    """Fetch SIDRA 9606 for AL and emit the current compiler's single-anchor artifact.

    Boundary rule:
    - the compiler receives the 102-row AL N6 municipal denominator panel;
    - the AL N6→N3 aggregate is retained only as an audit sidecar.
    """
    client = SidraClient()
    response = client.values(
        table_code="9606",
        periods=["2022"],
        variables=["93"],
        localities=localities,
        locality_level="N6",
        classifications={"86": ["95251"], "2": ["6794"], "287": ["100362"]},
    )
    if response.status_code >= 400 or not isinstance(response.payload, list):
        raise RuntimeError(f"SIDRA population request failed with HTTP {response.status_code}: {response.payload!r}")

    request = {
        "table_id": "9606",
        "periods": ["2022"],
        "variables": ["93"],
        "localities": localities,
        "locality_level": "N6",
        "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
    }

    header = response.payload[:1]
    rows = [
        row for row in response.payload[1:]
        if isinstance(row, dict)
        and str(row.get("D1C") or row.get("locality_id") or "").startswith("27")
    ]
    municipality_ids = sorted({str(row.get("D1C") or row.get("locality_id")) for row in rows})
    municipality_count = len(municipality_ids)
    if municipality_count != 102:
        raise RuntimeError(f"SIDRA AL N6 filter expected 102 municipalities, got {municipality_count}")

    municipal_request = {**request, "post_filter": "AL_N6_cod7_prefix_27"}
    municipal_payload = list(header) + rows
    municipal_facts = normalize_sidra_payload_to_facts(
        municipal_payload,
        table_id="9606",
        request_hash=content_hash(municipal_request),
        metadata_hash=content_hash({"table": "9606", "official": True, "state_panel": "AL", "support": "N6_municipality_year"}),
        chunk_request=municipal_request,
        unit_by_variable={"93": "persons"},
        fetched_at=(response.sidecar or {}).get("fetched_at"),
    )
    municipal_path = root / "processed" / "sidra" / "population_2022.parquet"
    municipal_path.parent.mkdir(parents=True, exist_ok=True)
    write_facts_parquet(municipal_facts, output_path=municipal_path)

    total = 0.0
    for row in rows:
        total += _parse_sidra_number(row.get("V", row.get("value")))

    aggregate_row = dict(rows[0])
    aggregate_row["NC"] = "3"
    aggregate_row["NN"] = "Unidade da Federação"
    aggregate_row["D1C"] = "27"
    aggregate_row["D1N"] = "Alagoas"
    aggregate_row["V"] = str(int(total)) if float(total).is_integer() else str(total)

    aggregate_request = {
        "table_id": "9606",
        "periods": ["2022"],
        "variables": ["93"],
        "localities": ["27"],
        "locality_level": "N3",
        "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
        "derived_from": {
            "source": "SIDRA_9606_N6_AL_municipal_total_rows",
            "aggregation": "additive_sum",
            "municipality_count": municipality_count,
            "municipality_ids": municipality_ids,
            "compile_artifact_path": str(municipal_path),
        },
    }
    aggregate_payload = list(header) + [aggregate_row]
    anchor_facts = normalize_sidra_payload_to_facts(
        aggregate_payload,
        table_id="9606",
        request_hash=content_hash(aggregate_request),
        metadata_hash=content_hash({"table": "9606", "official": True, "state_panel": "AL", "support": "N3_uf_year", "derived_from": "AL_N6_sum"}),
        chunk_request=aggregate_request,
        unit_by_variable={"93": "persons"},
        fetched_at=(response.sidecar or {}).get("fetched_at"),
    )
    numeric_anchor_facts = [
        fact for fact in anchor_facts
        if getattr(fact, "value_status", None) == "numeric"
        and str(getattr(fact, "locality_id", "")) == "27"
        and str(getattr(fact, "locality_level", "")) == "N3"
    ]
    if len(numeric_anchor_facts) != 1:
        raise RuntimeError(f"Expected exactly one aggregated AL N3 SIDRA anchor fact, got {len(numeric_anchor_facts)}")

    aggregate_path = root / "processed" / "sidra" / "population_2022_al_n3_aggregate_sidecar.parquet"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    write_facts_parquet(anchor_facts, output_path=aggregate_path)
    return municipal_path, bool(response.from_cache or response.status_code < 400), municipality_count


def _normalize_system(system: str, *, request: Any, output_path: Path, source_hash: str) -> None:
    if system == "SIM-DO":
        run_datasus_normalize_sim(input_path=request.processed_path, output_path=output_path, source_manifest_hash=source_hash)
    elif system == "SINASC":
        run_datasus_normalize_sinasc(input_path=request.processed_path, output_path=output_path, source_manifest_hash=source_hash)
    elif system == "CNES-ST":
        run_datasus_normalize_cnes(input_path=request.processed_path, output_path=output_path, source_manifest_hash=source_hash)
    elif system == "SIH-RD":
        run_datasus_normalize_sih(input_path=request.processed_path, output_path=output_path, source_manifest_hash=source_hash)
    else:
        raise ValueError(f"Unsupported DATASUS system: {system}")


def _concat_parquet_chunks(*, chunk_paths: list[Path], output_path: Path) -> None:
    existing = [path for path in chunk_paths if path.exists()]
    if not existing:
        raise RuntimeError(f"No normalized chunks exist for concatenation into {output_path}")
    frames = [pl.read_parquet(path) for path in existing]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) == 1:
        frames[0].write_parquet(output_path)
    else:
        pl.concat(frames, how="diagonal_relaxed").write_parquet(output_path)


def _normalize_system_requests(
    system: str,
    *,
    requests: list[Any],
    output_path: Path,
    source_hash: str,
) -> None:
    if not requests:
        raise RuntimeError(f"No successful DATASUS requests available for {system}")

    if len(requests) == 1:
        try:
            _normalize_system(system, request=requests[0], output_path=output_path, source_hash=source_hash)
        except Exception as exc:
            raise RuntimeError(
                f"Normalization failed for {system} input={requests[0].processed_path} "
                f"output={output_path}: {type(exc).__name__}: {exc}"
            ) from exc
        return

    chunk_dir = output_path.parent / "chunks" / system.replace("-", "_")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for request in requests:
        month = request.month_start
        # Annual (whole-year) requests carry no month; only zero-pad real months.
        try:
            month_token = f"{int(month):02d}"
        except (TypeError, ValueError):
            month_token = "all"
        chunk_path = chunk_dir / f"{request.year_start}_{month_token}_{request.request_hash[:12]}.parquet"
        try:
            _normalize_system(system, request=request, output_path=chunk_path, source_hash=source_hash)
        except Exception as exc:
            raise RuntimeError(
                f"Normalization failed for {system} month={request.month_start} "
                f"input={request.processed_path} output={chunk_path}: {type(exc).__name__}: {exc}"
            ) from exc
        chunk_paths.append(chunk_path)

    _concat_parquet_chunks(chunk_paths=chunk_paths, output_path=output_path)


def _table_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pl.read_parquet(path).to_dicts()


def _field_names(run_dir: Path) -> set[str]:
    rows = _table_rows(run_dir / "V_fields.parquet")
    names: set[str] = set()
    for row in rows:
        if row.get("name"):
            names.add(str(row["name"]))
        fid = str(row.get("field_id") or "")
        if fid and len(fid) != 64:
            names.add(fid)
    return names


def _forbidden_semantics(run_dir: Path) -> list[str]:
    forbidden = ("fixture", "synthetic", "explicit_smoke_municipality_crosswalk", "maceio support", "maceio smoke")
    hits: set[str] = set()
    for table_name in ("V_fields", "Q_tensor", "Warnings", "VariableDictionary", "E_DAG"):
        path = run_dir / f"{table_name}.parquet"
        for row in _table_rows(path):
            text = json.dumps(row, ensure_ascii=False, default=str).lower()
            for token in forbidden:
                if token in text:
                    hits.add(f"{table_name}:{token}")
    return sorted(hits)


def _support_pollution(run_dir: Path) -> list[str]:
    hits: list[str] = []
    # Columns holding materialized float vectors / scalar stats: a "270000" substring
    # inside a decimal expansion (e.g. 3484.2700000000004) is NOT a municipality-code
    # leak, so these are excluded from the geography-code scan to avoid false positives.
    numeric_value_columns = {"value_vector_json", "values_json", "value", "values", "statistic", "p_value", "q_value"}
    for table_name in ("V_fields", "Q_tensor", "Warnings", "VariableDictionary"):
        path = run_dir / f"{table_name}.parquet"
        for row in _table_rows(path):
            scanned = {k: v for k, v in row.items() if k not in numeric_value_columns}
            text = json.dumps(scanned, ensure_ascii=False, default=str)
            if "270000" in text:
                ident = row.get("field_id") or row.get("warning_id") or row.get("display_name") or "row"
                hits.append(f"{table_name}:{ident}")
    return hits[:50]


def _q_by_field(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("field_id")): row for row in _table_rows(run_dir / "Q_tensor.parquet") if row.get("field_id") is not None}


def _check_anomaly(run_dir: Path) -> dict[str, Any]:
    q = _q_by_field(run_dir)
    anomaly_field_id = "SINASCCongenitalAnomalyPrevalence"
    for field in _table_rows(run_dir / "V_fields.parquet"):
        if str(field.get("name") or "") == "SINASCCongenitalAnomalyPrevalence":
            anomaly_field_id = str(field.get("field_id") or anomaly_field_id)
            break
    row = q.get(anomaly_field_id) or {}
    n_events = row.get("n_events")
    n_denom = row.get("n_denom")
    rate = None
    try:
        if n_events is not None and n_denom:
            rate = float(n_events) / float(n_denom)
    except Exception:
        rate = None
    ok = rate is not None and 0.0 <= rate <= 0.20 and float(n_events or 0.0) < 10000.0
    return {"ok": ok, "n_events": n_events, "n_denom": n_denom, "rate": rate}


def _source_systems(overview: dict[str, Any]) -> list[str]:
    return sorted(str(x) for x in overview.get("source_reality", {}).get("source_systems", []))


def run_actual_state_panel(*, intent_name: str | None = None) -> dict[str, Any]:
    import os as _os
    # Generic env-driven scope: intent, panel root directory, and the DATASUS year
    # range are all parameters so the same real compiler path serves any UF/year
    # scope. Defaults reproduce the original single-year all-source Alagoas panel.
    intent_name = intent_name or _os.environ.get("PEGASUS_RUN_INTENT", "alagoas_2022_actual_allsource_state_panel.json")
    panel_dir = _os.environ.get("PEGASUS_RUN_ROOT", "alagoas_2022_allsource")
    years = _os.environ.get("PEGASUS_RUN_YEARS", "2022")
    run_subdir = _os.environ.get("PEGASUS_RUN_SUBDIR", "run")
    root = ROOT / "data" / "actual_state_panels" / panel_dir
    data_root = root / "sources"
    run_dir = root / run_subdir
    intent_path = ROOT / "config" / "intents" / intent_name
    # Honor the intent's own scope: which DATASUS systems to acquire and which
    # named fields are mandatory (the hardcoded set is only the all-source default).
    intent_doc: dict[str, Any] = {}
    if intent_path.exists():
        intent_doc = json.loads(intent_path.read_text(encoding="utf-8"))
    excluded = {str(s) for s in (intent_doc.get("exclude_systems") or [])}
    active_systems = tuple(s for s in DATASUS_SYSTEMS if s not in excluded)
    mandatory_fields = set(intent_doc.get("mandatory_fields") or MANDATORY_FIELD_NAMES)

    rscript = _find_rscript()
    payload: dict[str, Any] = {
        "classification": "failed",
        **_r_probe(rscript),
        "sidra_live_or_cache_available": False,
        "compile_attempted": False,
        "compile_source_mode": None,
        "run_dir": None,
        "municipality_count": 0,
        "output_validator_ok": False,
        "dashboard_read_only_ok": False,
        "q_tensor_nonempty": False,
        "efg_fields_nonempty": False,
        "efg_edges_nonempty": False,
        "source_artifact_reality": {},
        "source_systems_ok": False,
        "mandatory_fields_present": False,
        "missing_mandatory_fields": sorted(MANDATORY_FIELD_NAMES),
        "forbidden_semantics_present": [],
        "support_pollution_270000": [],
        "congenital_anomaly_check": {},
        "pirs_model_status": None,
        "pirs_hsic_status": None,
        "errors": [],
    }

    errors: list[str] = []
    try:
        if not intent_path.exists():
            raise FileNotFoundError(f"Intent does not exist: {intent_path}")
        required_r = ("rscript_found", "microdatasus_available", "read_dbc_available", "dplyr_available", "arrow_available", "jsonlite_available")
        if not all(payload[key] for key in required_r):
            payload["classification"] = "source_unavailable"
            raise RuntimeError(f"Rscript or required R packages unavailable: {payload.get('r_probe_error')}")

        client = MicrodatasusClient(
            config=DatasusConfig(
                rscript_path=str(rscript),
                r_library_path=str(R_LIBRARY),
                r_timeout_seconds=int(_os.environ.get("PEGASUS_DATASUS_R_TIMEOUT_SECONDS", 3600)),
                heartbeat_timeout_seconds=int(_os.environ.get("PEGASUS_DATASUS_HEARTBEAT_TIMEOUT_SECONDS", 300)),
                max_parallel_requests=int(_os.environ.get("PEGASUS_DATASUS_MAX_PARALLEL_REQUESTS", 8)),
            ),
            cache=DatasusCache(data_root / "cache"),
            data_root=data_root,
            manifest_root=data_root / "manifests",
        )

        # One global worker pool across all systems AND years (was serial per system).
        batches = client.fetch_systems(systems=list(active_systems), uf="AL", years=years)
        requests_by_system: dict[str, list[Any]] = {}
        for system in active_systems:
            batch = batches[system]
            if not batch.ok:
                payload["classification"] = "source_unavailable"
                raise RuntimeError(f"DATASUS acquisition failed for {system}: {batch.as_manifest()}")
            requests_by_system[system] = list(batch.requests)

        source_hash = content_hash({
            system: [request.request_hash for request in requests_by_system[system]]
            for system in sorted(requests_by_system)
        })
        normalized: dict[str, Path] = {}
        for system in active_systems:
            output_path = data_root / "normalized" / NORMALIZED_NAMES[system]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _normalize_system_requests(
                system,
                requests=requests_by_system[system],
                output_path=output_path,
                source_hash=source_hash,
            )
            normalized[system] = output_path

        sidra_facts, sidra_ok, municipality_count = _sidra_facts(root=data_root, localities=["all"])
        payload["sidra_live_or_cache_available"] = sidra_ok
        payload["municipality_count"] = municipality_count

        artifacts = [
            inspect_source_artifact(path=normalized[system], source_system=system, artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash)
            for system in active_systems
        ]
        artifacts.append(
            inspect_source_artifact(path=sidra_facts, source_system="SIDRA", artifact_role="normalized_facts", provenance_mode="materialized_external", source_manifest_hash=source_hash)
        )
        manifest = write_source_artifact_manifest(
            artifacts=artifacts,
            output_path=data_root / "source_artifacts.json",
            manifest_id=panel_dir,
        )

        payload["compile_attempted"] = True
        result = run_compile(
            intent_path=intent_path,
            run_dir=run_dir,
            data_root=root,
            source_manifest=manifest,
            require_materialized_external=True,
        )
        payload["run_dir"] = str(run_dir)
        payload["compile_result"] = {
            "status": result.get("status"),
            "run_id": result.get("run_id"),
            "run_dir": str(result.get("run_dir")),
            "compiler_stage_plan": result.get("compiler_stage_plan"),
            "race_bridge_plan": result.get("race_bridge_plan"),
            "cnes_sih": result.get("cnes_sih"),
            "population_tensor": result.get("population_tensor"),
            "autonomous_efg": result.get("autonomous_efg"),
        }

        validation = validate_output_bundle(run_dir=str(run_dir))
        payload["output_validator_ok"] = bool(validation.ok)
        if not validation.ok:
            errors.extend(validation.errors)

        overview = bundle_overview(run_dir=run_dir, limit=500)
        payload["dashboard_read_only_ok"] = overview["read_only"] is True and overview["validation_ok"] is True
        payload["compile_source_mode"] = overview["source_reality"].get("compile_source_mode")
        payload["source_artifact_reality"] = overview["source_reality"]
        expected_sources = sorted(set(active_systems) | {"SIDRA"})
        payload["source_systems_ok"] = _source_systems(overview) == expected_sources

        names = _field_names(run_dir)
        missing = sorted(set(mandatory_fields) - names)
        payload["missing_mandatory_fields"] = missing
        payload["mandatory_fields_present"] = not missing
        payload["efg_fields_nonempty"] = overview["efg"]["fields"]["row_count"] > 0
        payload["efg_edges_nonempty"] = overview["efg"]["edges"]["row_count"] > 0
        payload["q_tensor_nonempty"] = read_table_head(run_dir=run_dir, table_name="Q_tensor", limit=0)["row_count"] > 0
        payload["forbidden_semantics_present"] = _forbidden_semantics(run_dir)
        payload["support_pollution_270000"] = _support_pollution(run_dir)
        payload["congenital_anomaly_check"] = _check_anomaly(run_dir)

        repro = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
        stage_status = (((repro.get("telemetry") or {}).get("stage_status")) or {})
        payload["pirs_model_status"] = stage_status.get("pirs_model")
        payload["pirs_hsic_status"] = stage_status.get("pirs_hsic")

        success = (
            payload["compile_source_mode"] == "materialized_external"
            and payload["output_validator_ok"]
            and payload["dashboard_read_only_ok"]
            and payload["source_systems_ok"]
            and payload["mandatory_fields_present"]
            and payload["efg_fields_nonempty"]
            and payload["efg_edges_nonempty"]
            and payload["q_tensor_nonempty"]
            and not payload["forbidden_semantics_present"]
            and not payload["support_pollution_270000"]
            and payload["congenital_anomaly_check"].get("ok") is True
        )
        # PIRS/HSIC are not faked here. A skipped status keeps the run below full MSD-discovery grade.
        pirs_truthful = payload["pirs_model_status"] in {"success", "blocked"}
        hsic_truthful = payload["pirs_hsic_status"] in {"success", "blocked"}
        if success and pirs_truthful and hsic_truthful:
            payload["classification"] = "actual_allsource_state_panel_validated"
        elif success:
            payload["classification"] = "actual_allsource_state_panel_source_validated_pirs_hsic_pending"
        else:
            payload["classification"] = "source_partial_msd"
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.extend(traceback.format_exc().strip().splitlines()[-12:])
        if payload["classification"] == "failed" and payload["compile_attempted"]:
            payload["classification"] = "source_partial_msd"
    payload["errors"] = errors[:30]
    return payload


def main() -> int:
    payload = run_actual_state_panel()
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if payload["classification"] in {
        "actual_allsource_state_panel_validated",
        "actual_allsource_state_panel_source_validated_pirs_hsic_pending",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
