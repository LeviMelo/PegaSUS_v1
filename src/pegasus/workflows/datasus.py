from __future__ import annotations

from pathlib import Path
from typing import Any

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


def run_datasus_ingest(
    *,
    system: str,
    uf: str,
    years: str,
    dry_run: bool,
) -> dict[str, Any]:
    cfg_payload = load_datasus_config()
    cfg = DatasusConfig.from_mapping(cfg_payload)
    manifests = build_datasus_manifests(system=system, uf=uf, years=years, config=cfg_payload)

    cache = DatasusCache()
    planned = []
    executed = []
    blocked = False
    failed = False

    for manifest in manifests:
        planned_path = write_request_manifest(manifest)
        planned.append((manifest, planned_path))

        if dry_run:
            continue

        result = fetch_datasus_chunk(
            manifest,
            config=cfg,
            cache=cache,
            timeout_seconds=cfg.r_timeout_seconds,
            heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
        )
        executed_path = write_request_manifest(result)
        executed.append((result, executed_path))

        if result.status == "blocked":
            blocked = True
        elif result.status != "success":
            failed = True

    return {
        "planned": planned,
        "executed": executed,
        "blocked": blocked,
        "failed": failed,
    }


def run_datasus_profile(*, manifest: str | Path) -> dict[str, Any]:
    request = read_request_manifest(manifest)
    raw_path = Path(request.raw_path)
    processed_path = Path(request.processed_path)

    if not raw_path.exists() or not processed_path.exists():
        return {"status": "blocked", "reason": "raw_or_processed_artifact_missing", "request": request}

    raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
    processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
    compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

    raw_profile = profile_table(raw_path, output_path=raw_profile_path)
    processed_profile = profile_table(processed_path, output_path=processed_profile_path)
    comparison = compare_profiles(raw_profile, processed_profile, output_path=compare_path)

    return {
        "status": "success",
        "request": request,
        "raw_profile_path": raw_profile_path,
        "processed_profile_path": processed_profile_path,
        "compare_path": compare_path,
        "raw_profile": raw_profile,
        "processed_profile": processed_profile,
        "comparison": comparison,
    }


def run_datasus_normalize_sim(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    return normalize_sim_do_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
