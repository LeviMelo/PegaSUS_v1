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

DATASUS_BRIDGE_CONTRACT_VERSION = "datasus_r_bridge_v3_utf8_sanitized_raw_canonical_plus_microdatasus_sidecar"
from pegasus.datasus.manifests import utc_now


def datasus_dependency_unavailable(*, dependency: str, detail: str) -> str:
    """Return the stable actionable diagnostic for unavailable live DATASUS."""
    return json.dumps(
        {
            "source": "DATASUS",
            "status": "unavailable",
            "required_dependency": dependency,
            "detail": detail,
            "action": "install/configure Rscript and microdatasus, or use fixture/cached mode",
        },
        ensure_ascii=True,
        sort_keys=True,
    )


@dataclass(frozen=True)
class DatasusConfig:
    rscript_path: str = "Rscript"
    r_library_path: str | None = None
    r_timeout_seconds: int = 7200
    heartbeat_timeout_seconds: int = 900

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasusConfig":
        r_library_path = payload.get("r_library_path")
        return cls(
            rscript_path=str(payload.get("rscript_path", "Rscript")),
            r_library_path=None if r_library_path in {None, ""} else str(r_library_path),
            r_timeout_seconds=int(payload.get("r_timeout_seconds", 7200)),
            heartbeat_timeout_seconds=int(payload.get("heartbeat_timeout_seconds", 900)),
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
    if manifest_path.exists() and raw_path.exists() and processed_path.exists():
        try:
            cached_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_payload.get("status") != "success":
                raise ValueError("cached R manifest is not successful")
            if cached_payload.get("processing_contract_version") != DATASUS_BRIDGE_CONTRACT_VERSION:
                raise ValueError("cached R manifest uses an obsolete or non-UTF8-sanitized DATASUS bridge contract")
            return _finish(
                request,
                started=started,
                status="cached",
                exit_code=0,
                error_message=None,
                updates={
                    "raw_sha256": sha256_file(raw_path),
                    "processed_sha256": sha256_file(processed_path),
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
        actual_raw_path = Path(payload.get("raw_path", request.raw_path))
        actual_processed_path = Path(payload.get("processed_path", request.processed_path))

        if not actual_raw_path.exists() or not actual_processed_path.exists():
            return _finish(
                request,
                started=started,
                status="failed",
                exit_code=50,
                error_message="R manifest exists but raw or processed artifact is missing.",
            )

        return _finish(
            request,
            started=started,
            status="success",
            exit_code=0,
            error_message=None,
            updates={
                "raw_path": str(actual_raw_path),
                "processed_path": str(actual_processed_path),
                "raw_sha256": sha256_file(actual_raw_path),
                "processed_sha256": sha256_file(actual_processed_path),
                "row_counts": payload.get("row_counts", {}),
                "column_lists": payload.get("column_lists", {}),
                "r_version": payload.get("r_version"),
                "microdatasus_version": payload.get("microdatasus_version"),
                "read_dbc_version": payload.get("read_dbc_version"),
            },
        )
    except Exception as exc:
        return _finish(
            request,
            started=started,
            status="failed",
            exit_code=50,
            error_message=f"Invalid R manifest: {exc}",
        )
