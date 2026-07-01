from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_ROOT = ROOT / ".codex-tmp" / "msd_convergence_slice1_backups" / STAMP
TOUCHED: list[str] = []


def rel(path: str) -> Path:
    return ROOT / path


def backup(path: Path) -> None:
    if not path.exists():
        return
    target = BACKUP_ROOT / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)


def read(path: str) -> str:
    p = rel(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = p.read_text(encoding="utf-8")
        if old == text:
            return
        backup(p)
    p.write_text(text, encoding="utf-8", newline="\n")
    TOUCHED.append(path)


def patch(path: str, fn) -> None:
    old = read(path)
    new = fn(old)
    if old != new:
        write(path, new)


def replace_once(text: str, old: str, new: str, *, path: str) -> str:
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def replace_all(text: str, old: str, new: str, *, path: str) -> str:
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new)


def regex_replace_once(text: str, pattern: str, replacement: str, *, path: str, flags: int = re.S) -> str:
    compiled = re.compile(pattern, flags)
    new, n = compiled.subn(lambda _m: replacement, text, count=1)
    if n != 1:
        raise RuntimeError(f"Regex patch anchor not found in {path}: {pattern[:160]!r}")
    return new


STATE_PANEL = r'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import polars as pl


AL_UF_PREFIX = "27"


@dataclass(frozen=True)
class MunicipalitySupport:
    municipalities: list[str]
    invalid_cod6: list[str]
    uf_prefix: str

    @property
    def valid_count(self) -> int:
        return len(self.municipalities)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_cod6)

    def as_support_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "municipalities": self.municipalities,
            "municipality_count_valid": self.valid_count,
            "invalid_municipality_cod6_count": self.invalid_count,
            "uf_prefix": self.uf_prefix,
            "municipality_code_policy": "datasus_cod6_valid_municipality_only",
        }
        if self.invalid_cod6:
            payload["invalid_municipality_cod6"] = self.invalid_cod6
            payload["invalid_municipality_code_policy"] = "excluded_from_municipality_support_not_from_raw_audit"
        return payload


def normalize_cod6(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[:6]
    return digits or None


def is_valid_datasus_municipality_cod6(value: Any, *, uf_prefix: str = AL_UF_PREFIX) -> bool:
    code = normalize_cod6(value)
    if code is None:
        return False
    if len(code) != 6:
        return False
    if not code.startswith(str(uf_prefix)):
        return False
    if code.endswith("0000"):
        return False
    if code[-4:] == "0000":
        return False
    return True


def clean_datasus_municipalities(values: Iterable[Any], *, uf_prefix: str = AL_UF_PREFIX) -> tuple[list[str], list[str]]:
    valid: set[str] = set()
    invalid: set[str] = set()
    for value in values:
        code = normalize_cod6(value)
        if code is None:
            continue
        if is_valid_datasus_municipality_cod6(code, uf_prefix=uf_prefix):
            valid.add(code)
        else:
            invalid.add(code)
    return sorted(valid), sorted(invalid)


def municipality_support_from_values(values: Iterable[Any], *, uf_prefix: str = AL_UF_PREFIX) -> MunicipalitySupport:
    valid, invalid = clean_datasus_municipalities(values, uf_prefix=uf_prefix)
    return MunicipalitySupport(valid, invalid, uf_prefix)


def filter_valid_cod6_frame(df: pl.DataFrame, column: str, *, uf_prefix: str = AL_UF_PREFIX) -> pl.DataFrame:
    if column not in df.columns:
        return df
    return df.filter(
        pl.col(column)
        .cast(pl.Utf8)
        .map_elements(lambda value: is_valid_datasus_municipality_cod6(value, uf_prefix=uf_prefix), return_dtype=pl.Boolean)
    )
'''.lstrip()


ACTUAL_STATE_PANEL_RUNTIME = r'''
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
from pegasus.workflows.acquire.cnes_sih import run_datasus_normalize_cnes, run_datasus_normalize_sih
from pegasus.workflows.compile import run_compile
from pegasus.workflows.acquire.datasus import run_datasus_normalize_sim
from pegasus.workflows.acquire.sinasc import run_datasus_normalize_sinasc

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


def _sidra_facts(*, root: Path, localities: list[str]) -> tuple[Path, bool, int]:
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
    municipality_count = len({str(row.get("D1C") or row.get("locality_id")) for row in rows})
    if municipality_count != 102:
        raise RuntimeError(f"SIDRA AL N6 filter expected 102 municipalities, got {municipality_count}")
    municipal_request = {**request, "post_filter": "AL_N6_cod7_prefix_27"}
    facts = normalize_sidra_payload_to_facts(
        list(header) + rows,
        table_id="9606",
        request_hash=content_hash(municipal_request),
        metadata_hash=content_hash({"table": "9606", "official": True, "state_panel": "AL", "support": "N6_municipality_year"}),
        chunk_request=municipal_request,
        unit_by_variable={"93": "persons"},
        fetched_at=(response.sidecar or {}).get("fetched_at"),
    )
    path = root / "processed" / "sidra" / "population_2022.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_facts_parquet(facts, output_path=path)
    return path, bool(response.from_cache or response.status_code < 400), municipality_count


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
    for table_name in ("V_fields", "Q_tensor", "Warnings", "VariableDictionary"):
        path = run_dir / f"{table_name}.parquet"
        for row in _table_rows(path):
            text = json.dumps(row, ensure_ascii=False, default=str)
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


def run_actual_state_panel(*, intent_name: str = "alagoas_2022_actual_allsource_state_panel.json") -> dict[str, Any]:
    root = ROOT / "data" / "actual_state_panels" / "alagoas_2022_allsource"
    data_root = root / "sources"
    run_dir = root / "run"
    intent_path = ROOT / "config" / "intents" / intent_name

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
                r_timeout_seconds=3600,
                heartbeat_timeout_seconds=600,
            ),
            cache=DatasusCache(data_root / "cache"),
            data_root=data_root,
            manifest_root=data_root / "manifests",
        )

        batches: dict[str, Any] = {}
        requests: dict[str, Any] = {}
        for system in DATASUS_SYSTEMS:
            batch = client.fetch(system=system, uf="AL", years="2022")
            batches[system] = batch
            if not batch.ok:
                payload["classification"] = "source_unavailable"
                raise RuntimeError(f"DATASUS acquisition failed for {system}: {batch.as_manifest()}")
            requests[system] = batch.requests[0]

        source_hash = content_hash({system: requests[system].request_hash for system in sorted(requests)})
        normalized: dict[str, Path] = {}
        for system in DATASUS_SYSTEMS:
            output_path = data_root / "normalized" / NORMALIZED_NAMES[system]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _normalize_system(system, request=requests[system], output_path=output_path, source_hash=source_hash)
            normalized[system] = output_path

        sidra_facts, sidra_ok, municipality_count = _sidra_facts(root=data_root, localities=["all"])
        payload["sidra_live_or_cache_available"] = sidra_ok
        payload["municipality_count"] = municipality_count

        artifacts = [
            inspect_source_artifact(path=normalized["SIM-DO"], source_system="SIM-DO", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash),
            inspect_source_artifact(path=normalized["SINASC"], source_system="SINASC", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash),
            inspect_source_artifact(path=normalized["CNES-ST"], source_system="CNES-ST", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash),
            inspect_source_artifact(path=normalized["SIH-RD"], source_system="SIH-RD", artifact_role="processed_events", provenance_mode="materialized_external", source_manifest_hash=source_hash),
            inspect_source_artifact(path=sidra_facts, source_system="SIDRA", artifact_role="normalized_facts", provenance_mode="materialized_external", source_manifest_hash=source_hash),
        ]
        manifest = write_source_artifact_manifest(
            artifacts=artifacts,
            output_path=data_root / "source_artifacts.json",
            manifest_id="alagoas_2022_actual_allsource_state_panel",
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
        payload["compile_result"] = result

        validation = validate_output_bundle(run_dir=str(run_dir))
        payload["output_validator_ok"] = bool(validation.ok)
        if not validation.ok:
            errors.extend(validation.errors)

        overview = bundle_overview(run_dir=run_dir, limit=500)
        payload["dashboard_read_only_ok"] = overview["read_only"] is True and overview["validation_ok"] is True
        payload["compile_source_mode"] = overview["source_reality"].get("compile_source_mode")
        payload["source_artifact_reality"] = overview["source_reality"]
        payload["source_systems_ok"] = _source_systems(overview) == EXPECTED_SOURCE_SYSTEMS

        names = _field_names(run_dir)
        missing = sorted(MANDATORY_FIELD_NAMES - names)
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
'''.lstrip()


AUDIT_ENTRY = '''from actual_state_panel_runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
'''


INTENT = {
    "budget": "standard",
    "context_policy": [
        "actual_source_state_panel",
        "official_sidra_anchor",
        "sinasc_maternal_child_compile",
        "live_birth_denominator_linkage",
        "include_cnes_sih",
        "run_pirs",
        "run_hsic",
    ],
    "execution_scale": "state",
    "geography": {"codes": [], "level": "municipality", "uf": ["AL"]},
    "health_seeds": ["all_cause_mortality", "icd_chapter", "icd_block", "hospitalization", "capacity"],
    "mandatory_fields": [
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
    ],
    "population_mode": "official_sidra_anchor",
    "race_tensor_mode": "decoupled",
    "system_weights": {"SIDRA": 1.0, "SIM-DO": 1.0, "SINASC": 1.0, "CNES-ST": 1.0, "SIH-RD": 1.0},
    "time": {"start_year": 2022, "end_year": 2022, "start_month": None, "end_month": None},
}


TESTS = r'''
from __future__ import annotations

import pytest

from pegasus.datasus.sinasc_normalize import decode_anomaly_flag, normalize_anomaly_icd
from pegasus.geo.state_panel import clean_datasus_municipalities, is_valid_datasus_municipality_cod6


def test_alagoas_cod6_support_excludes_state_pseudocode() -> None:
    valid, invalid = clean_datasus_municipalities(["270000", "270010", "270430", None, "foo"], uf_prefix="27")
    assert valid == ["270010", "270430"]
    assert "270000" in invalid
    assert not is_valid_datasus_municipality_cod6("270000", uf_prefix="27")


def test_sinasc_anomaly_decoder_is_not_inverted() -> None:
    absent, absent_state = decode_anomaly_flag("1")
    present, present_state = decode_anomaly_flag("2")
    assert absent is False
    assert present is True
    assert absent_state == "valid_absent"
    assert present_state == "valid_present"

    assert normalize_anomaly_icd(None, absent) == (None, "absent", False)
    assert normalize_anomaly_icd("Q359", absent) == ("Q359", "valid_q_anomaly", True)
    assert normalize_anomaly_icd("Z000", None) == ("Z000", "valid_non_q_not_anomaly", False)
    assert normalize_anomaly_icd("not-an-icd", None) == (None, "invalid", False)


def test_sinasc_anomaly_present_flag_survives_missing_code() -> None:
    assert normalize_anomaly_icd(None, True) == (None, "flag_present_code_missing", True)
    assert normalize_anomaly_icd("bad", True) == (None, "flag_present_invalid_code", True)
'''.lstrip()


def write_static_files() -> None:
    write("src/pegasus/geo/state_panel.py", STATE_PANEL)
    write("scripts/dev/audits/actual_state_panel_runtime.py", ACTUAL_STATE_PANEL_RUNTIME)
    write("scripts/dev/audits/audit_actual_data_state_panel_alagoas_2022.py", AUDIT_ENTRY)
    write("config/intents/alagoas_2022_actual_allsource_state_panel.json", json.dumps(INTENT, indent=2, ensure_ascii=False) + "\n")
    write("tests/integration/test_msd_state_panel_semantics.py", TESTS)


def patch_sinasc_normalize() -> None:
    path = "src/pegasus/datasus/sinasc_normalize.py"

    def fn(text: str) -> str:
        text = regex_replace_once(
            text,
            r"def normalize_anomaly_icd\(value: Any, anomaly_flag: bool \| None\) -> tuple\[str \| None, str, bool\]:.*?\n\ndef normalize_sinasc_record",
            dedent('''
            def normalize_anomaly_icd(value: Any, anomaly_flag: bool | None) -> tuple[str | None, str, bool]:
                """Normalize SINASC congenital-anomaly ICD marks without inverting IDANOMAL.

                IDANOMAL is declaration state: 1 = no anomaly, 2 = anomaly present, 9/missing = unknown.
                CODANOMAL is a diagnostic mark and only Q* ICD-like codes are congenital-anomaly codes by
                topology. Non-Q diagnostic strings are preserved but do not become anomaly numerators unless
                the explicit anomaly-present flag is true.
                """
                text = _clean(value)
                flag_present = anomaly_flag is True
                if text is None:
                    if flag_present:
                        return None, "flag_present_code_missing", True
                    if anomaly_flag is False:
                        return None, "absent", False
                    return None, "unknown", False

                token = re.split(r"[;|,\\s]+", text.upper().strip())[0]
                match = ICD_LIKE.match(token)
                if not match:
                    if flag_present:
                        return None, "flag_present_invalid_code", True
                    return None, "invalid", False

                code = match.group(0)
                if code.startswith("Q"):
                    return code, "valid_q_anomaly", True
                if flag_present:
                    return code, "valid_non_q_with_present_flag", True
                return code, "valid_non_q_not_anomaly", False


            def normalize_sinasc_record''').lstrip(),
            path=path,
        )

        anchor = "    out.parent.mkdir(parents=True, exist_ok=True)\n    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out)\n    return {\n"
        replacement = """    out.parent.mkdir(parents=True, exist_ok=True)
    valid_rows = sum(1 for row in rows if row["record_state"] == "valid")
    anomaly_rows = sum(1 for row in rows if row["congenital_anomaly_flag"] is True)
    if valid_rows and anomaly_rows / float(valid_rows) > 0.20:
        raise ValueError(
            "SINASC congenital anomaly numerator exceeds 20% of valid births; "
            "this usually indicates inverted IDANOMAL/CODANOMAL semantics."
        )
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(out)
    return {
"""
        text = replace_once(text, anchor, replacement, path=path)
        text = replace_once(text, '"valid_rows": sum(1 for row in rows if row["record_state"] == "valid"),', '"valid_rows": valid_rows,', path=path)
        text = replace_once(text, '"anomaly_rows": sum(1 for row in rows if row["congenital_anomaly_flag"] is True),', '"anomaly_rows": anomaly_rows,', path=path)
        return text

    patch(path, fn)


def patch_full_inference() -> None:
    for path in ("src/pegasus/datasus/sih_normalize.py", "src/pegasus/datasus/cnes_normalize.py"):
        def fn(text: str, path=path) -> str:
            return text.replace("pl.DataFrame(rows).write_parquet(out)", "pl.DataFrame(rows, infer_schema_length=None).write_parquet(out)")
        patch(path, fn)


def patch_maternal_child() -> None:
    path = "src/pegasus/she/maternal_child.py"

    helper = dedent('''

    def _congenital_anomaly_count(df: pl.DataFrame) -> int:
        if "congenital_anomaly_flag" not in df.columns:
            return 0
        return int(df.filter(pl.col("congenital_anomaly_flag") == True).height)  # noqa: E712


    def _assert_plausible_anomaly_rate(*, births_total: int, congenital_anomaly_births: int) -> None:
        if births_total <= 0:
            return
        rate = congenital_anomaly_births / float(births_total)
        if rate > 0.20:
            raise ValueError(
                f"SINASC congenital anomaly count is implausibly high: "
                f"{congenital_anomaly_births}/{births_total} ({rate:.3f})."
            )
    ''').rstrip()

    def fn(text: str) -> str:
        if "from pegasus.geo.state_panel import clean_datasus_municipalities" not in text:
            text = text.replace("import polars as pl\n", "import polars as pl\n\nfrom pegasus.geo.state_panel import clean_datasus_municipalities\n", 1)
        if "def _congenital_anomaly_count" not in text:
            text = text.replace("def _state_not_valid_count", helper + "\n\n\ndef _state_not_valid_count", 1)
        text = replace_once(
            text,
            "    race_missing = 0\n",
            "    municipalities, invalid_municipalities = clean_datasus_municipalities(municipalities, uf_prefix=\"27\")\n    congenital_anomaly_births = _congenital_anomaly_count(valid)\n    _assert_plausible_anomaly_rate(births_total=int(valid.height), congenital_anomaly_births=congenital_anomaly_births)\n\n    race_missing = 0\n",
            path=path,
        )
        text = replace_once(
            text,
            '        congenital_anomaly_births=_bool_count(valid, "congenital_anomaly_flag"),',
            "        congenital_anomaly_births=congenital_anomaly_births,",
            path=path,
        )
        return text

    patch(path, fn)


def patch_maternal_child_linkage() -> None:
    path = "src/pegasus/she/maternal_child_linkage.py"

    helper = dedent('''

    def _congenital_anomaly_count(df: pl.DataFrame) -> int:
        if "congenital_anomaly_flag" not in df.columns:
            return 0
        return int(df.filter(pl.col("congenital_anomaly_flag") == True).height)  # noqa: E712


    def _assert_plausible_anomaly_rate(*, births_total: int, congenital_anomaly_births: int) -> None:
        if births_total <= 0:
            return
        rate = congenital_anomaly_births / float(births_total)
        if rate > 0.20:
            raise ValueError(
                f"SINASC congenital anomaly count is implausibly high: "
                f"{congenital_anomaly_births}/{births_total} ({rate:.3f})."
            )
    ''').rstrip()

    def fn(text: str) -> str:
        if "from pegasus.geo.state_panel import clean_datasus_municipalities" not in text:
            text = text.replace("import polars as pl\n", "import polars as pl\n\nfrom pegasus.geo.state_panel import clean_datasus_municipalities\n", 1)
        if "def _congenital_anomaly_count" not in text:
            text = text.replace("def summarize_maternal_child_linkage", helper + "\n\n\ndef summarize_maternal_child_linkage", 1)
        text = replace_once(
            text,
            "    municipalities = sorted(set(municipalities))\n\n    cod7s = [str(municipality_ibge_cod7)] if municipality_ibge_cod7 else []\n",
            "    municipalities = sorted(set(municipalities))\n    municipalities, invalid_municipalities = clean_datasus_municipalities(municipalities, uf_prefix=\"27\")\n\n    cod7s = [str(municipality_ibge_cod7)] if municipality_ibge_cod7 else []\n",
            path=path,
        )
        text = replace_once(
            text,
            "    postneonatal = infant.filter((pl.col(\"age_days\") >= 28) & (pl.col(\"age_days\") < 365.25)) if infant.height else infant\n\n    return MaternalChildLinkedSummary(\n",
            "    postneonatal = infant.filter((pl.col(\"age_days\") >= 28) & (pl.col(\"age_days\") < 365.25)) if infant.height else infant\n    congenital_anomaly_births = _congenital_anomaly_count(sinasc_valid)\n    _assert_plausible_anomaly_rate(births_total=int(sinasc_valid.height), congenital_anomaly_births=congenital_anomaly_births)\n\n    return MaternalChildLinkedSummary(\n",
            path=path,
        )
        text = replace_once(
            text,
            '        congenital_anomaly_births=_bool_count(sinasc_valid, "congenital_anomaly_flag"),',
            "        congenital_anomaly_births=congenital_anomaly_births,",
            path=path,
        )
        return text

    patch(path, fn)


def patch_maternal_child_compile_attach() -> None:
    path = "src/pegasus/output/maternal_child_compile_attach.py"

    support_alignment = dedent('''
    def _support_alignment(summary: MaternalChildLinkedSummary) -> dict[str, Any]:
        if summary.municipalities_ibge_cod7:
            crosswalk = "datasus_cod6_to_ibge_cod7_explicit_crosswalk"
            denominator_cod7 = summary.municipalities_ibge_cod7
            reason = "municipality_year_exact_after_datasus_cod6_to_ibge_cod7_crosswalk"
        else:
            crosswalk = "native_datasus_cod6_state_panel"
            denominator_cod7 = []
            reason = "state_panel_native_datasus_cod6_support_without_single_municipality_ibge7_filter"

        return {
            "aligned": True,
            "reason": reason,
            "numerator_municipalities_cod6": summary.municipalities_cod6,
            "numerator_municipalities_ibge_cod7": summary.municipalities_ibge_cod7,
            "denominator_municipalities_ibge_cod7": denominator_cod7,
            "numerator_years": summary.years,
            "denominator_years": summary.years,
            "crosswalk": crosswalk,
            "support_kind": "state_panel_cod6_year" if not summary.municipalities_ibge_cod7 else "municipality_year_cod6_cod7",
        }


    def _lineage''').lstrip()

    warning_rows = dedent('''
    def _warning_rows(summary: MaternalChildLinkedSummary) -> list[dict[str, Any]]:
        return [
            {
                "warning_id": "maternal_child_state_panel_linkage",
                "field_id": "run",
                "severity": "info",
                "message": _json({
                    "births_total": summary.births_total,
                    "infant_deaths": summary.infant_deaths,
                    "neonatal_deaths": summary.neonatal_deaths,
                    "postneonatal_deaths": summary.postneonatal_deaths,
                    "support_alignment": _support_alignment(summary),
                    "source_reality": "materialized_external",
                }),
                "created_at": _now(),
                "inherited_from": None,
            }
        ]


    def _build_fields''').lstrip()

    cov_helpers = dedent('''
    def _support_payload(field: dict[str, Any]) -> dict[str, Any]:
        raw = field.get("support_json")
        if not isinstance(raw, str) or not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


    def _support_cov_s(field: dict[str, Any]) -> float:
        payload = _support_payload(field)
        geography = payload.get("geography")
        if isinstance(geography, dict):
            for key in ("municipality_cod6", "municipality_ibge_cod7"):
                values = geography.get(key)
                if isinstance(values, list):
                    return float(len(values))
        values = payload.get("municipalities")
        if isinstance(values, list):
            return float(len(values))
        return 1.0


    def _support_cov_t(field: dict[str, Any]) -> float:
        payload = _support_payload(field)
        time = payload.get("time")
        if isinstance(time, dict):
            years = time.get("years")
            if isinstance(years, list):
                return float(len(years))
        years = payload.get("years")
        if isinstance(years, list):
            return float(len(years))
        return 1.0


    def _q_row''').lstrip()

    def fn(text: str) -> str:
        text = regex_replace_once(text, r"def _support_alignment\(summary: MaternalChildLinkedSummary\) -> dict\[str, Any\]:.*?\n\ndef _lineage", support_alignment, path=path)
        text = regex_replace_once(text, r"def _warning_rows\(summary: MaternalChildLinkedSummary\) -> list\[dict\[str, Any\]\]:.*?\n\ndef _build_fields", warning_rows, path=path)
        if "def _support_cov_s" not in text:
            text = text.replace("def _q_row", cov_helpers, 1)
        text = text.replace('"cov_S": 1.0,', '"cov_S": _support_cov_s(field),')
        text = text.replace('"cov_T": 1.0,', '"cov_T": _support_cov_t(field),')
        text = text.replace("slice3b_compile_linkage", "maternal_child_state_panel_linkage")
        text = text.replace("slice3b_support_aligned_maternal_child", "state_panel_support_aligned_maternal_child")
        text = text.replace('"fixture_small_n", ', "")
        text = text.replace('"fixture_small_n"', '"state_panel_descriptive_quarantine"')
        text = text.replace("explicit_smoke_municipality_crosswalk", "native_datasus_cod6_state_panel")
        return text

    patch(path, fn)


def patch_sim_efg_bundle() -> None:
    path = "src/pegasus/output/sim_efg_bundle.py"

    def fn(text: str) -> str:
        if "from pegasus.geo.state_panel import clean_datasus_municipalities" not in text:
            text = text.replace("import polars as pl\n", "import polars as pl\n\nfrom pegasus.geo.state_panel import clean_datasus_municipalities\n", 1)
        old = '''    out = {
        "support": "municipality_year",
        "years": years,
        "municipalities": municipalities,
        "cov_S": float(len(municipalities)),
        "cov_T": float(len(years)),
        "missingness": float(missingness),
        "denom_fragility": float(denom_fragility),
    }
'''
        new = '''    valid_municipalities, invalid_municipalities = clean_datasus_municipalities(municipalities, uf_prefix="27")
    out = {
        "support": "municipality_year",
        "years": years,
        "municipalities": valid_municipalities,
        "cov_S": float(len(valid_municipalities)),
        "cov_T": float(len(years)),
        "missingness": float(missingness),
        "denom_fragility": float(denom_fragility),
        "municipality_code_policy": "datasus_cod6_valid_municipality_only",
    }
    if invalid_municipalities:
        out["invalid_municipality_cod6"] = invalid_municipalities
        out["invalid_municipality_cod6_count"] = float(len(invalid_municipalities))
'''
        text = replace_once(text, old, new, path=path)
        return text

    patch(path, fn)


def patch_cnes_sih_bundle_support() -> None:
    path = "src/pegasus/output/cnes_sih_efg_bundle.py"

    def fn(text: str) -> str:
        if "from pegasus.geo.state_panel import clean_datasus_municipalities" not in text:
            text = text.replace("from typing import Any\n", "from typing import Any\n\nfrom pegasus.geo.state_panel import clean_datasus_municipalities\n", 1)
        # Make support constructors clean COD6 lists when the module exposes the common local support object.
        text = text.replace(
            '"municipality_cod6": municipalities_cod6,',
            '"municipality_cod6": clean_datasus_municipalities(municipalities_cod6, uf_prefix="27")[0],',
        )
        text = text.replace(
            '"municipalities_cod6": municipalities_cod6,',
            '"municipalities_cod6": clean_datasus_municipalities(municipalities_cod6, uf_prefix="27")[0],',
        )
        return text

    patch(path, fn)


def main() -> None:
    write_static_files()
    patch_sinasc_normalize()
    patch_full_inference()
    patch_maternal_child()
    patch_maternal_child_linkage()
    patch_maternal_child_compile_attach()
    patch_sim_efg_bundle()
    patch_cnes_sih_bundle_support()

    print("MSD convergence slice 1 applied.")
    print(f"Backups written under: {BACKUP_ROOT}")
    print("Touched files:")
    for item in sorted(set(TOUCHED)):
        print(f"  - {item}")


if __name__ == "__main__":
    main()
