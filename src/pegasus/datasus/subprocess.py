from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pegasus.core.hashing import sha256_file
from pegasus.core.schemas import DATASUSRequestManifest
from pegasus.datasus.cache import DatasusCache


class DatasusConfig:
    def __init__(self, rscript_path: str = "Rscript") -> None:
        self.rscript_path = rscript_path


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
    rscript = shutil.which(config.rscript_path) or config.rscript_path
    if shutil.which(config.rscript_path) is None and not Path(config.rscript_path).exists():
        return request.model_copy(update={
            "status": "blocked",
            "exit_code": 41,
            "error_message": f"Rscript not found: {config.rscript_path}",
        })

    script = Path(__file__).parent / "r_scripts" / "fetch_process_microdatasus.R"
    if not script.exists():
        return request.model_copy(update={
            "status": "blocked",
            "exit_code": 11,
            "error_message": f"R bridge script missing: {script}",
        })

    out_dir = Path(request.raw_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = Path(request.heartbeat_path)
    stdout_path = Path(request.stdout_path)
    stderr_path = Path(request.stderr_path)

    command = [
        rscript,
        str(script),
        "--system", request.system,
        "--uf", request.uf,
        "--year-start", str(request.year_start),
        "--year-end", str(request.year_end),
        "--out-dir", str(out_dir),
    ]
    if request.month_start is not None:
        command.extend(["--month-start", str(request.month_start)])
    if request.month_end is not None:
        command.extend(["--month-end", str(request.month_end)])

    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            elapsed = time.time() - started
            if elapsed > timeout_seconds or _heartbeat_stale(heartbeat_path, heartbeat_timeout_seconds):
                _kill_process_tree(process)
                return request.model_copy(update={
                    "status": "timeout",
                    "exit_code": 41,
                    "error_message": "R subprocess timed out or heartbeat became stale.",
                })
            time.sleep(1.0)

    manifest_path = out_dir / "manifest.json"
    if process.returncode != 0:
        return request.model_copy(update={
            "status": "failed",
            "exit_code": int(process.returncode or 60),
            "error_message": f"R subprocess failed with exit code {process.returncode}.",
        })

    if not manifest_path.exists():
        return request.model_copy(update={
            "status": "failed",
            "exit_code": 50,
            "error_message": "R subprocess returned success but manifest.json is missing.",
        })

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_path = Path(payload.get("raw_path", request.raw_path))
        processed_path = Path(payload.get("processed_path", request.processed_path))
        if not raw_path.exists() or not processed_path.exists():
            return request.model_copy(update={
                "status": "failed",
                "exit_code": 50,
                "error_message": "R manifest exists but raw or processed artifact is missing.",
            })

        return request.model_copy(update={
            "status": "success",
            "exit_code": 0,
            "error_message": None,
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "raw_sha256": sha256_file(raw_path),
            "processed_sha256": sha256_file(processed_path),
        })
    except Exception as exc:
        return request.model_copy(update={
            "status": "failed",
            "exit_code": 50,
            "error_message": f"Invalid R manifest: {exc}",
        })
