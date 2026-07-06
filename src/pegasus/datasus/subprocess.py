from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import sha256_file
from pegasus.core.schemas import DATASUSRequestManifest
from pegasus.datasus.cache import DatasusCache

# v4 drops the redundant per-chunk sidecars (raw.rds + microdatasus_processed.parquet) and
# writes processed.parquet with zstd. The CONSUMED artifact (processed.parquet, raw-coded, all
# columns) is schema-identical to v3, so a v3 chunk stays cache-valid after its raw.rds is GC'd —
# hence the accepted-versions set rather than a single-version equality gate.
DATASUS_BRIDGE_CONTRACT_VERSION = "datasus_r_bridge_v4_zstd_canonical_only"
ACCEPTED_DATASUS_CONTRACT_VERSIONS = frozenset({
    "datasus_r_bridge_v3_utf8_sanitized_raw_canonical_plus_microdatasus_sidecar",
    DATASUS_BRIDGE_CONTRACT_VERSION,
})
from pegasus.datasus.manifests import utc_now


def _prune_success_ancillary(request: DATASUSRequestManifest) -> None:
    """Delete per-chunk debug ephemera (stdout/stderr/heartbeat) once a chunk is durably
    materialized. They exist only to diagnose a *failing* R subprocess; on success they are
    tens of thousands of tiny files of pure slack. manifest.json + processed.parquet are kept."""
    for attr in ("stdout_path", "stderr_path", "heartbeat_path"):
        try:
            Path(getattr(request, attr)).unlink(missing_ok=True)
        except OSError:
            pass


def _sha256_sidecar_path(processed_path: Path) -> Path:
    """The digest sidecar co-located with a chunk's processed.parquet."""
    return processed_path.with_name(processed_path.name + ".sha256")


def _write_processed_sha256_sidecar(processed_path: Path) -> str:
    """Hash ``processed_path`` once (it is hot in cache right after being written) and
    persist the digest to a sidecar so warm cache hits can skip the re-read. Returns the
    digest; sidecar write failures are non-fatal (the digest is still returned)."""
    digest = sha256_file(processed_path)
    try:
        _sha256_sidecar_path(processed_path).write_text(digest, encoding="utf-8")
    except OSError:
        pass
    return digest


def _cached_processed_sha256(processed_path: Path) -> str:
    """Return the processed.parquet digest for a warm cache hit without re-reading the
    (GB-scale) parquet: prefer the sidecar; fall back to a one-time hash + sidecar write
    for legacy chunks that predate the sidecar."""
    sidecar = _sha256_sidecar_path(processed_path)
    try:
        cached = sidecar.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    except OSError:
        pass
    return _write_processed_sha256_sidecar(processed_path)


def datasus_dependency_unavailable(*, dependency: str, detail: str) -> str:
    """Return the stable actionable diagnostic for unavailable live DATASUS."""
    return json.dumps(
        {
            "source": "DATASUS",
            "status": "unavailable",
            "required_dependency": dependency,
            "detail": detail,
            "action": "install/configure Rscript and microdatasus, or provide a materialized source artifact manifest",
        },
        ensure_ascii=True,
        sort_keys=True,
    )


@dataclass(frozen=True)
class DatasusConfig:
    rscript_path: str = "Rscript"
    r_library_path: str | None = None
    r_timeout_seconds: int = 7200
    heartbeat_timeout_seconds: int = 300
    max_parallel_requests: int = 8

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasusConfig":
        r_library_path = payload.get("r_library_path")
        r_timeout_raw = os.environ.get("PEGASUS_DATASUS_R_TIMEOUT_SECONDS", payload.get("r_timeout_seconds", 7200))
        heartbeat_timeout_raw = os.environ.get(
            "PEGASUS_DATASUS_HEARTBEAT_TIMEOUT_SECONDS",
            payload.get("heartbeat_timeout_seconds", 900),
        )
        parallel_raw = os.environ.get(
            "PEGASUS_DATASUS_MAX_PARALLEL_REQUESTS",
            payload.get("max_parallel_requests", 4),
        )
        return cls(
            rscript_path=str(payload.get("rscript_path", "Rscript")),
            r_library_path=None if r_library_path in {None, ""} else str(r_library_path),
            r_timeout_seconds=int(r_timeout_raw),
            heartbeat_timeout_seconds=int(heartbeat_timeout_raw),
            max_parallel_requests=max(1, int(parallel_raw)),
        )

    @classmethod
    def from_file(cls, path: str | Path = "config/datasus.yaml") -> "DatasusConfig":
        data = load_yaml(path)
        return cls.from_mapping(data.get("datasus", data))


def _duration(started: float) -> float:
    return round(time.time() - started, 6)


def _finish(
    request: DATASUSRequestManifest,
    *,
    started: float,
    status: str,
    exit_code: int,
    error_message: str | None,
    updates: dict[str, Any] | None = None,
) -> DATASUSRequestManifest:
    payload = {
        "status": status,
        "exit_code": exit_code,
        "error_message": error_message,
        "ended_at": utc_now(),
        "duration_seconds": _duration(started),
    }
    if updates:
        payload.update(updates)
    return request.model_copy(update=payload)


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()


def _heartbeat_stale(path: Path, timeout_seconds: int) -> bool:
    if not path.exists():
        return False
    return time.time() - path.stat().st_mtime > timeout_seconds


def fetch_datasus_chunk(
    request: DATASUSRequestManifest,
    *,
    config: DatasusConfig,
    cache: DatasusCache,
    timeout_seconds: int,
    heartbeat_timeout_seconds: int,
) -> DATASUSRequestManifest:
    started = time.time()
    request = request.model_copy(update={"started_at": utc_now(), "rscript_path": config.rscript_path})

    rscript = shutil.which(config.rscript_path) or config.rscript_path
    if shutil.which(config.rscript_path) is None and not Path(config.rscript_path).exists():
        return _finish(
            request,
            started=started,
            status="blocked",
            exit_code=41,
            error_message=datasus_dependency_unavailable(
                dependency="Rscript / microdatasus",
                detail=f"Rscript not found: {config.rscript_path}",
            ),
        )

    script = Path(__file__).parent / "r_scripts" / "fetch_process_microdatasus.R"
    if not script.exists():
        return _finish(
            request,
            started=started,
            status="blocked",
            exit_code=11,
            error_message=f"R bridge script missing: {script}",
        )

    raw_path = Path(request.raw_path)
    processed_path = Path(request.processed_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_path.parent / "manifest.json"
    # Cache validity is keyed on the CONSUMED artifact (processed.parquet) + its manifest, NOT on
    # the raw.rds sidecar — so a chunk whose redundant raw.rds has been GC'd still hits cache and
    # is never re-fetched.
    if manifest_path.exists() and processed_path.exists():
        try:
            cached_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_payload.get("status") != "success":
                raise ValueError("cached R manifest is not successful")
            if cached_payload.get("processing_contract_version") not in ACCEPTED_DATASUS_CONTRACT_VERSIONS:
                raise ValueError("cached R manifest uses an obsolete or non-UTF8-sanitized DATASUS bridge contract")
            return _finish(
                request,
                started=started,
                status="cached",
                exit_code=0,
                error_message=None,
                updates={
                    "raw_sha256": sha256_file(raw_path) if raw_path.exists() else "",
                    # Warm cache hit: read the processed_sha256 from the sidecar written on the
                    # original materialization instead of re-reading the (GB-scale) processed.parquet
                    # end-to-end on every hit. Legacy chunks (pre-sidecar) hash once, then persist it.
                    "processed_sha256": _cached_processed_sha256(processed_path),
                    "row_counts": cached_payload.get("row_counts", {}),
                    "column_lists": cached_payload.get("column_lists", {}),
                    "r_version": cached_payload.get("r_version"),
                    "microdatasus_version": cached_payload.get("microdatasus_version"),
                    "read_dbc_version": cached_payload.get("read_dbc_version"),
                },
            )
        except (OSError, ValueError, TypeError):
            pass

    heartbeat_path = Path(request.heartbeat_path)
    stdout_path = Path(request.stdout_path)
    stderr_path = Path(request.stderr_path)
    heartbeat_path.unlink(missing_ok=True)

    command = [
        rscript,
        "--vanilla",
        str(script),
        "--system",
        request.system,
        "--uf",
        request.uf,
        "--year-start",
        str(request.year_start),
        "--year-end",
        str(request.year_end),
        "--raw-path",
        str(raw_path),
        "--processed-path",
        str(processed_path),
        "--out-dir",
        str(raw_path.parent),
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    if request.month_start is not None:
        command.extend(["--month-start", str(request.month_start)])
    if request.month_end is not None:
        command.extend(["--month-end", str(request.month_end)])

    cache.write_request(request.request_hash, request.model_dump(mode="json"))

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        environment = os.environ.copy()
        if config.r_library_path:
            environment["R_LIBS_USER"] = config.r_library_path
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        while process.poll() is None:
            elapsed = time.time() - started
            if elapsed > timeout_seconds or _heartbeat_stale(heartbeat_path, heartbeat_timeout_seconds):
                _kill_process_tree(process)
                return _finish(
                    request,
                    started=started,
                    status="timeout",
                    exit_code=41,
                    error_message="R subprocess timed out or heartbeat became stale.",
                )
            time.sleep(1.0)

    if process.returncode != 0:
        return _finish(
            request,
            started=started,
            status="failed",
            exit_code=int(process.returncode or 60),
            error_message=f"R subprocess failed with exit code {process.returncode}.",
        )

    if not manifest_path.exists():
        return _finish(
            request,
            started=started,
            status="failed",
            exit_code=50,
            error_message="R subprocess returned success but manifest.json is missing.",
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_value = payload.get("raw_path", request.raw_path)
        actual_raw_path = Path(raw_value) if raw_value else None
        actual_processed_path = Path(payload.get("processed_path", request.processed_path))

        # Only the consumed artifact (processed.parquet) is required; raw.rds is optional and,
        # under the v4 bridge, not written at all.
        if not actual_processed_path.exists():
            return _finish(
                request,
                started=started,
                status="failed",
                exit_code=50,
                error_message="R manifest exists but the processed.parquet artifact is missing.",
            )

        finished = _finish(
            request,
            started=started,
            status="success",
            exit_code=0,
            error_message=None,
            updates={
                "raw_path": str(actual_raw_path) if actual_raw_path is not None else request.raw_path,
                "processed_path": str(actual_processed_path),
                "raw_sha256": sha256_file(actual_raw_path) if actual_raw_path is not None and actual_raw_path.exists() else "",
                # Fresh materialization: hash the just-written (hot) parquet once and persist the
                # digest to a sidecar so subsequent warm cache hits never re-read the file.
                "processed_sha256": _write_processed_sha256_sidecar(actual_processed_path),
                "row_counts": payload.get("row_counts", {}),
                "column_lists": payload.get("column_lists", {}),
                "r_version": payload.get("r_version"),
                "microdatasus_version": payload.get("microdatasus_version"),
                "read_dbc_version": payload.get("read_dbc_version"),
            },
        )
        _prune_success_ancillary(finished)
        return finished
    except Exception as exc:
        return _finish(
            request,
            started=started,
            status="failed",
            exit_code=50,
            error_message=f"Invalid R manifest: {exc}",
        )
