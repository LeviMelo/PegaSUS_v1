from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.dashboard.read_only import bundle_overview, read_table_head
from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.datasus.subprocess import DatasusConfig
from pegasus.output.validate import validate_output_bundle
from pegasus.sidra.api import SidraClient
from pegasus.sidra.facts import write_facts_parquet
from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
from pegasus.source_artifacts.contracts import inspect_source_artifact, write_source_artifact_manifest
from pegasus.workflows.compile import run_compile
from pegasus.workflows.datasus import run_datasus_normalize_sim
from pegasus.workflows.sinasc import run_datasus_normalize_sinasc

ROOT = Path(__file__).resolve().parents[3]
R_LIBRARY = ROOT / ".r-library"
MANDATORY_FIELDS = [
    "SIMDeathsAll", "SIMCrudeMortalitySIDRAOfficial", "SIDRAPopulationTotalAnchor",
    "SINASCLiveBirthsAll", "SINASCCrudeBirthRateSIDRAOfficial",
    "SINASCLowBirthWeightPrevalence", "SINASCPrematurityPrevalence",
    "SINASCCongenitalAnomalyPrevalence", "SIMInfantMortalitySINASCBirths",
    "SIMNeonatalMortalitySINASCBirths", "SIMPostNeonatalMortalitySINASCBirths",
]


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
    result = {"rscript_found": bool(rscript), "rscript_path": rscript, "microdatasus_available": False,
              "read_dbc_available": False, "dplyr_available": False, "r_probe_error": None}
    if not rscript:
        return result
    library_literal = json.dumps(str(R_LIBRARY).replace("\\", "/"))
    expression = f".libPaths(c({library_literal},.libPaths()));cat(requireNamespace('microdatasus',quietly=TRUE),requireNamespace('read.dbc',quietly=TRUE),requireNamespace('dplyr',quietly=TRUE),sep='|')"
    process = subprocess.run([rscript, "--vanilla", "-e", expression], capture_output=True, text=True, check=False, timeout=60)
    values = process.stdout.strip().split("|")
    if len(values) == 3:
        result.update(microdatasus_available=values[0] == "TRUE", read_dbc_available=values[1] == "TRUE", dplyr_available=values[2] == "TRUE")
    if process.returncode != 0:
        result["r_probe_error"] = process.stderr.strip()[:1000] or f"R package probe exited {process.returncode}"
    elif not all(result[key] for key in ("microdatasus_available", "read_dbc_available", "dplyr_available")):
        result["r_probe_error"] = "Required packages are absent from the vanilla R library path."
    return result


def _sidra_facts(*, root: Path, localities: list[str]) -> tuple[Path, bool, int]:
    client = SidraClient()
    response = client.values(table_code="9606", periods=["2022"], variables=["93"], localities=localities,
                             locality_level="N6", classifications={"86": ["95251"], "2": ["6794"], "287": ["100362"]})
    if response.status_code >= 400 or not isinstance(response.payload, list):
        raise RuntimeError(f"SIDRA population request failed with HTTP {response.status_code}: {response.payload!r}")
    request = {"table_id": "9606", "periods": ["2022"], "variables": ["93"], "localities": localities,
               "locality_level": "N6", "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]}}
    facts = normalize_sidra_payload_to_facts(response.payload, table_id="9606", request_hash=content_hash(request),
                                             metadata_hash=content_hash({"table": "9606", "official": True}),
                                             chunk_request=request, unit_by_variable={"93": "persons"},
                                             fetched_at=(response.sidecar or {}).get("fetched_at"))
    path = root / "processed" / "sidra" / "population_2022.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_facts_parquet(facts, output_path=path)
    municipality_count = len({str(row.get("D1C")) for row in response.payload[1:] if isinstance(row, dict) and row.get("D1C")})
    return path, bool(response.from_cache or response.status_code < 400), municipality_count


def _field_names(overview: dict[str, Any]) -> set[str]:
    rows = overview.get("efg", {}).get("fields", {}).get("rows", [])
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("name"):
            names.add(str(row["name"]))
        field_id = str(row.get("field_id") or "")
        if field_id and len(field_id) != 64:
            names.add(field_id)
    return names


def _fixture_semantics(overview: dict[str, Any]) -> list[str]:
    rows = overview.get("efg", {}).get("fields", {}).get("rows", [])
    contaminated: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = " ".join(str(row.get(key, "")) for key in ("name", "source", "provenance", "warnings")).lower()
        if "fixture" in text:
            contaminated.append(str(row.get("name") or row.get("field_id") or "unknown"))
    return sorted(set(contaminated))


def classify_actual_smoke(payload: dict[str, Any], *, grid: bool) -> str:
    required = (
        payload.get("compile_source_mode") != "fixture_only",
        payload.get("output_validator_ok") is True,
        payload.get("dashboard_read_only_ok") is True,
        payload.get("mandatory_fields_present") is True,
        payload.get("efg_fields_nonempty") is True,
        payload.get("efg_edges_nonempty") is True,
        payload.get("q_tensor_nonempty") is True,
        not payload.get("fixture_semantics_present"),
    )
    if grid and int(payload.get("municipality_count") or 0) <= 1:
        return "source_partial"
    return "actual_data_validated" if all(required) else "source_partial"


def run_actual_smoke(*, intent_name: str, grid: bool) -> dict[str, Any]:
    root = ROOT / "data" / "actual_smokes" / ("alagoas_2022" if grid else "maceio_2022")
    run_dir = root / "run"
    errors: list[str] = []
    rscript = _find_rscript()
    payload: dict[str, Any] = {
        "classification": "failed", **_r_probe(rscript), "sidra_live_or_cache_available": False,
        "compile_attempted": False, "compile_source_mode": None, "run_dir": None, "municipality_count": 0,
        "output_validator_ok": False, "dashboard_read_only_ok": False, "mandatory_fields_present": False,
        "missing_mandatory_fields": list(MANDATORY_FIELDS), "q_tensor_nonempty": False,
        "efg_fields_nonempty": False, "efg_edges_nonempty": False, "source_artifact_reality": {}, "errors": errors,
        "fixture_semantics_present": [],
    }
    try:
        if not all(payload[key] for key in ("rscript_found", "microdatasus_available", "read_dbc_available", "dplyr_available")):
            payload["classification"] = "source_unavailable"
            raise RuntimeError(f"Rscript or required DATASUS R packages are unavailable: {payload.get('r_probe_error')}")
        data_root = root / "sources"
        client = MicrodatasusClient(config=DatasusConfig(rscript_path=str(rscript), r_library_path=str(R_LIBRARY), r_timeout_seconds=3600, heartbeat_timeout_seconds=300),
                                    cache=DatasusCache(data_root / "cache"), data_root=data_root,
                                    manifest_root=data_root / "manifests")
        sim = client.fetch(system="SIM-DO", uf="AL", years="2022")
        sinasc = client.fetch(system="SINASC", uf="AL", years="2022")
        if not sim.ok or not sinasc.ok:
            payload["classification"] = "source_unavailable"
            raise RuntimeError(f"DATASUS acquisition failed: SIM={sim.as_manifest()} SINASC={sinasc.as_manifest()}")
        sim_request, sinasc_request = sim.requests[0], sinasc.requests[0]
        sim_events = data_root / "normalized" / "sim_events.parquet"
        sinasc_events = data_root / "normalized" / "sinasc_events.parquet"
        source_hash = content_hash({"sim": sim_request.request_hash, "sinasc": sinasc_request.request_hash})
        run_datasus_normalize_sim(input_path=sim_request.processed_path, output_path=sim_events, source_manifest_hash=source_hash)
        run_datasus_normalize_sinasc(input_path=sinasc_request.processed_path, output_path=sinasc_events, source_manifest_hash=source_hash)
        localities = ["2704302"] if not grid else ["all"]
        sidra_facts, sidra_ok, municipality_count = _sidra_facts(root=data_root, localities=localities)
        payload["sidra_live_or_cache_available"] = sidra_ok
        payload["municipality_count"] = municipality_count
        artifacts = [
            inspect_source_artifact(path=sim_events, source_system="SIM-DO", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash),
            inspect_source_artifact(path=sinasc_events, source_system="SINASC", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash),
            inspect_source_artifact(path=sidra_facts, source_system="SIDRA", artifact_role="normalized_facts", provenance_mode="materialized_external", source_manifest_hash=source_hash),
        ]
        manifest = write_source_artifact_manifest(artifacts=artifacts, output_path=data_root / "source_artifacts.json",
                                                  manifest_id=f"actual_smoke_{'alagoas' if grid else 'maceio'}_2022")
        payload["compile_attempted"] = True
        result = run_compile(intent_path=ROOT / "config" / "intents" / intent_name, run_dir=run_dir, data_root=root,
                             source_manifest=manifest, require_materialized_external=True)
        payload["run_dir"] = str(run_dir)
        payload["output_validator_ok"] = bool(validate_output_bundle(run_dir=str(run_dir)).ok)
        overview = bundle_overview(run_dir=run_dir, limit=500)
        payload["dashboard_read_only_ok"] = overview["read_only"] is True and overview["validation_ok"] is True
        payload["compile_source_mode"] = overview["source_reality"].get("compile_source_mode")
        payload["source_artifact_reality"] = overview["source_reality"]
        fields = _field_names(overview)
        missing = sorted(set(MANDATORY_FIELDS) - fields)
        payload["missing_mandatory_fields"] = missing
        payload["mandatory_fields_present"] = not missing
        payload["efg_fields_nonempty"] = overview["efg"]["fields"]["row_count"] > 0
        payload["efg_edges_nonempty"] = overview["efg"]["edges"]["row_count"] > 0
        payload["q_tensor_nonempty"] = read_table_head(run_dir=run_dir, table_name="Q_tensor", limit=0)["row_count"] > 0
        payload["fixture_semantics_present"] = _fixture_semantics(overview)
        payload["classification"] = classify_actual_smoke(payload, grid=grid)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        if payload["classification"] == "failed" and payload["compile_attempted"]:
            payload["classification"] = "source_partial"
        errors.extend(traceback.format_exc().strip().splitlines()[-8:])
    payload["errors"] = errors[:20]
    return payload


def main(*, intent_name: str, grid: bool) -> int:
    payload = run_actual_smoke(intent_name=intent_name, grid=grid)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if payload["classification"] == "actual_data_validated" else 1
