from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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


_ALL_UFS: tuple[str, ...] = (
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG",
    "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
)


def _resolve_ufs(uf: str | list[str] | tuple[str, ...]) -> list[str]:
    """Resolve a UF selector to a concrete list. ``ALL``/``BR``/``*`` → all 27 UFs; a
    comma-separated string or a list are expanded; a single UF stays a singleton."""
    if isinstance(uf, (list, tuple)):
        return [str(u).strip() for u in uf if str(u).strip()]
    text = str(uf).strip()
    if text.upper() in {"ALL", "BR", "*"}:
        return list(_ALL_UFS)
    if "," in text:
        return [u.strip() for u in text.split(",") if u.strip()]
    return [text]


def run_datasus_ingest(
    *,
    system: str,
    uf: str | list[str] | tuple[str, ...],
    years: str,
    dry_run: bool,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Acquire DATASUS chunks, fetching every (UF × period) manifest **concurrently**.

    The manifest builder deliberately splits into many small per-(UF, year[, month])
    chunks so throughput comes from download parallelism, not a few huge R processes
    (manifests.py). ``uf`` may be a single UF, a list, a comma string, or ``"ALL"`` (all
    27 UFs) — national acquisition is then a single wide parallel batch across all
    ``UF × period`` chunks. ``max_workers`` overrides the config's ``max_parallel_requests``
    (DATASUS FTP has no formal rate limit, so this is bounded only by CPU/RAM). Cached
    chunks return instantly, so re-runs only fetch what is missing.
    """
    cfg_payload = load_datasus_config()
    cfg = DatasusConfig.from_mapping(cfg_payload)

    manifests = []
    for one_uf in _resolve_ufs(uf):
        manifests.extend(build_datasus_manifests(system=system, uf=one_uf, years=years, config=cfg_payload))

    cache = DatasusCache()
    planned = [(m, write_request_manifest(m)) for m in manifests]
    if dry_run or not manifests:
        return {"planned": planned, "executed": [], "blocked": False, "failed": False}

    workers = int(max_workers) if max_workers else int(getattr(cfg, "max_parallel_requests", 8) or 8)
    workers = max(1, min(workers, len(manifests)))

    def _fetch(manifest):
        return fetch_datasus_chunk(
            manifest,
            config=cfg,
            cache=cache,
            timeout_seconds=cfg.r_timeout_seconds,
            heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
        )

    executed: list = []
    blocked = False
    failed = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed({pool.submit(_fetch, m) for m in manifests}):
            result = future.result()
            executed.append((result, write_request_manifest(result)))
            if result.status == "blocked":
                blocked = True
            elif result.status not in {"success", "cached"}:
                failed = True

    return {"planned": planned, "executed": executed, "blocked": blocked, "failed": failed}


def run_datasus_profile(*, manifest: str | Path) -> dict[str, Any]:
    request = read_request_manifest(manifest)
    raw_path = Path(request.raw_path)
    processed_path = Path(request.processed_path)

    # processed.parquet is the consumed, load-bearing artifact; raw.rds is an optional legacy
    # sidecar (not written by the v4 bridge). The audit profiles what is present.
    if not processed_path.exists():
        return {"status": "blocked", "reason": "processed_artifact_missing", "request": request}

    raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
    processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
    compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

    processed_profile = profile_table(processed_path, output_path=processed_profile_path)
    if not raw_path.exists():
        return {
            "status": "success",
            "request": request,
            "raw_profile_path": None,
            "processed_profile_path": processed_profile_path,
            "compare_path": None,
            "raw_profile": None,
            "processed_profile": processed_profile,
            "comparison": None,
        }

    raw_profile = profile_table(raw_path, output_path=raw_profile_path)
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
