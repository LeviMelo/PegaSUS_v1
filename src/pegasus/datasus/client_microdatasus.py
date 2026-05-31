from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from pegasus.core.hashing import sha256_file, sha256_json
from pegasus.core.manifests import utc_now_iso
from pegasus.datasus.io import read_table, write_json, write_parquet
from pegasus.datasus.profile import profile_dataframe
from pegasus.datasus.schema_compare import compare_profiles


DATASUS_PROCESS_FUNCTIONS = {
    "SIM-DO": "process_sim",
    "SINASC": "process_sinasc",
    "SIH-RD": "process_sih",
    "CNES-ST": "process_cnes",
}

DATASUS_SYSTEM_ALIASES = {
    "SIM": "SIM-DO",
    "SIM-DO": "SIM-DO",
    "DO": "SIM-DO",
    "SINASC": "SINASC",
    "SIH": "SIH-RD",
    "SIH-RD": "SIH-RD",
    "CNES": "CNES-ST",
    "CNES-ST": "CNES-ST",
}


class MicrodatasusRequest(BaseModel):
    system: Literal["SIM-DO", "SINASC", "SIH-RD", "CNES-ST"]
    uf: str
    year_start: int
    year_end: int
    month_start: int | None = None
    month_end: int | None = None

    def normalized_uf(self) -> str:
        return self.uf.upper().strip()

    def period_label(self) -> str:
        ms = "NA" if self.month_start is None else f"{self.month_start:02d}"
        me = "NA" if self.month_end is None else f"{self.month_end:02d}"
        return f"{self.year_start}_{ms}__{self.year_end}_{me}"

    def hash_payload(self) -> dict:
        return {
            "backend": "microdatasus",
            "system": self.system,
            "information_system": self.system,
            "uf": self.normalized_uf(),
            "year_start": self.year_start,
            "year_end": self.year_end,
            "month_start": self.month_start,
            "month_end": self.month_end,
            "fetch_function": "fetch_datasus",
            "process_function": DATASUS_PROCESS_FUNCTIONS[self.system],
        }

    def request_hash(self) -> str:
        return sha256_json(self.hash_payload())


class MicrodatasusArtifactPaths(BaseModel):
    request_hash: str
    work_dir: str

    raw_csv: str
    processed_csv: str
    r_manifest: str
    heartbeat: str
    stdout_log: str
    stderr_log: str
    python_invocation: str

    raw_parquet: str
    processed_parquet: str
    raw_profile_json: str
    processed_profile_json: str
    comparison_json: str
    final_manifest_json: str


class MicrodatasusIngestManifest(BaseModel):
    backend: str = "microdatasus"
    status: Literal["ok", "raw_ok_process_error", "error"]
    request: MicrodatasusRequest
    request_hash: str

    rscript_path: str
    r_script_path: str
    command: list[str]
    cwd: str
    created_at: str
    started_at: str
    completed_at: str | None = None
    elapsed_seconds: float | None = None
    exit_code: int | None = None

    paths: MicrodatasusArtifactPaths

    r_manifest: dict | None = None
    error_message: str | None = None

    raw_rows: int | None = None
    raw_cols: int | None = None
    processed_rows: int | None = None
    processed_cols: int | None = None
    raw_columns: list[str] = Field(default_factory=list)
    processed_columns: list[str] = Field(default_factory=list)

    raw_sha256: str | None = None
    processed_sha256: str | None = None
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None

    warnings: list[str] = Field(default_factory=list)


def prepare_microdatasus_work_dir_for_run(paths: MicrodatasusArtifactPaths) -> None:
    """Remove stale transient R workdir artifacts before a non-cache run.

    This does not delete immutable raw/processed Parquet artifacts. It only
    cleans the R bridge work directory so stale heartbeat/raw CSV state cannot
    be misread as belonging to the current attempt.
    """
    transient_paths = [
        paths.raw_csv,
        paths.processed_csv,
        paths.r_manifest,
        paths.heartbeat,
    ]

    for value in transient_paths:
        path = Path(value)
        if path.exists():
            path.unlink()


def normalize_datasus_system(system: str) -> str:
    key = system.upper().strip()
    if key not in DATASUS_SYSTEM_ALIASES:
        raise ValueError(f"Unsupported DATASUS system: {system}")
    return DATASUS_SYSTEM_ALIASES[key]


def build_microdatasus_paths(
    *,
    root: str | Path,
    request: MicrodatasusRequest,
) -> MicrodatasusArtifactPaths:
    root = Path(root).resolve()
    request_hash = request.request_hash()
    system = request.system
    uf = request.normalized_uf()
    period = request.period_label()

    work_dir = (
        root
        / "data"
        / "cache"
        / "datasus"
        / "microdatasus"
        / system
        / f"uf={uf}"
        / f"period={period}"
        / request_hash
    )

    raw_dir = (
        root
        / "data"
        / "raw"
        / "datasus"
        / system
        / f"uf={uf}"
        / f"period={period}"
        / request_hash
    )

    processed_dir = (
        root
        / "data"
        / "processed"
        / "datasus"
        / system
        / f"uf={uf}"
        / f"period={period}"
        / request_hash
    )

    metadata_dir = (
        root
        / "data"
        / "metadata"
        / "datasus"
        / system
        / f"uf={uf}"
        / f"period={period}"
        / request_hash
    )

    manifest_dir = root / "data" / "manifests" / "datasus" / system
    diagnostics_dir = root / "data" / "diagnostics" / "datasus" / system / request_hash

    for directory in (
        work_dir,
        raw_dir,
        processed_dir,
        metadata_dir,
        manifest_dir,
        diagnostics_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return MicrodatasusArtifactPaths(
        request_hash=request_hash,
        work_dir=str(work_dir),
        raw_csv=str(work_dir / "raw.csv"),
        processed_csv=str(work_dir / "processed.csv"),
        r_manifest=str(work_dir / "manifest.json"),
        heartbeat=str(work_dir / "heartbeat.json"),
        stdout_log=str(diagnostics_dir / "r_stdout.log"),
        stderr_log=str(diagnostics_dir / "r_stderr.log"),
        python_invocation=str(diagnostics_dir / "python_invocation.json"),
        raw_parquet=str(raw_dir / "raw.parquet"),
        processed_parquet=str(processed_dir / "processed.parquet"),
        raw_profile_json=str(metadata_dir / "raw_profile.json"),
        processed_profile_json=str(metadata_dir / "processed_profile.json"),
        comparison_json=str(metadata_dir / "raw_processed_comparison.json"),
        final_manifest_json=str(manifest_dir / f"{request_hash}.json"),
    )


def build_r_command(
    *,
    request: MicrodatasusRequest,
    paths: MicrodatasusArtifactPaths,
    rscript_path: str,
    r_script_path: str | Path,
) -> list[str]:
    command = [
        rscript_path,
        str(Path(r_script_path).resolve()),
        "--system",
        request.system,
        "--uf",
        request.normalized_uf(),
        "--year-start",
        str(request.year_start),
        "--year-end",
        str(request.year_end),
        "--out-dir",
        paths.work_dir,
    ]

    if request.month_start is not None:
        command.extend(["--month-start", str(request.month_start)])

    if request.month_end is not None:
        command.extend(["--month-end", str(request.month_end)])

    return command


def _heartbeat_is_stale(
    path: Path,
    *,
    process_start_wall_time: float,
    threshold_seconds: int,
) -> bool:
    if not path.exists():
        return False

    mtime = path.stat().st_mtime

    # Ignore heartbeat files from previous attempts. They are stale as files,
    # but they are not evidence that the current R process is stuck.
    if mtime < process_start_wall_time:
        return False

    age = time.time() - mtime
    return age > threshold_seconds


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        process.kill()


def run_r_microdatasus(
    *,
    command: list[str],
    cwd: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    heartbeat_path: str | Path,
    timeout_seconds: int,
    heartbeat_timeout_seconds: int,
) -> int:
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    heartbeat_path = Path(heartbeat_path)

    creationflags = 0
    preexec_fn = None

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        preexec_fn = os.setsid

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )

        started = time.perf_counter()
        process_start_wall_time = time.time()

        while True:
            exit_code = process.poll()

            if exit_code is not None:
                return exit_code

            elapsed = time.perf_counter() - started

            if elapsed > timeout_seconds:
                _terminate_process(process)
                return 41

            if _heartbeat_is_stale(
                heartbeat_path,
                process_start_wall_time=process_start_wall_time,
                threshold_seconds=heartbeat_timeout_seconds,
            ):
                _terminate_process(process)
                return 41

            time.sleep(2)


def load_json_if_exists(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def r_manifest_error_message(r_manifest: dict | None) -> str | None:
    if not r_manifest:
        return None

    value = r_manifest.get("error_message")
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def stderr_tail(path: str | Path, n: int = 2000) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text[-n:].strip()
    return text or None


def materialize_raw_only(
    *,
    request: MicrodatasusRequest,
    paths: MicrodatasusArtifactPaths,
) -> pl.DataFrame:
    raw_df = read_table(paths.raw_csv)
    write_parquet(raw_df, paths.raw_parquet)

    raw_profile = profile_dataframe(
        raw_df,
        source_system=request.system,
        artifact_kind="raw",
    )
    write_json(paths.raw_profile_json, raw_profile)

    return raw_df


def materialize_raw_and_processed(
    *,
    request: MicrodatasusRequest,
    paths: MicrodatasusArtifactPaths,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw_df = read_table(paths.raw_csv)
    processed_df = read_table(paths.processed_csv)

    write_parquet(raw_df, paths.raw_parquet)
    write_parquet(processed_df, paths.processed_parquet)

    raw_profile = profile_dataframe(
        raw_df,
        source_system=request.system,
        artifact_kind="raw",
    )
    processed_profile = profile_dataframe(
        processed_df,
        source_system=request.system,
        artifact_kind="processed",
    )
    comparison = compare_profiles(raw_profile, processed_profile)

    write_json(paths.raw_profile_json, raw_profile)
    write_json(paths.processed_profile_json, processed_profile)
    write_json(paths.comparison_json, comparison)

    return raw_df, processed_df


def ingest_microdatasus(
    *,
    root: str | Path,
    request: MicrodatasusRequest,
    rscript_path: str = "Rscript",
    timeout_seconds: int = 7200,
    heartbeat_timeout_seconds: int = 900,
    use_cache: bool = True,
) -> MicrodatasusIngestManifest:
    root = Path(root).resolve()
    paths = build_microdatasus_paths(root=root, request=request)

    r_script_path = (
        root / "src" / "pegasus" / "datasus" / "r_scripts" / "fetch_process_microdatasus.R"
    )

    command = build_r_command(
        request=request,
        paths=paths,
        rscript_path=rscript_path,
        r_script_path=r_script_path,
    )

    started_at = utc_now_iso()
    start = time.perf_counter()

    if use_cache and Path(paths.raw_parquet).exists():
        raw_df = pl.read_parquet(paths.raw_parquet)
        processed_exists = Path(paths.processed_parquet).exists()

        processed_df = (
            pl.read_parquet(paths.processed_parquet)
            if processed_exists
            else None
        )

        status: Literal["ok", "raw_ok_process_error"] = (
            "ok" if processed_exists else "raw_ok_process_error"
        )

        manifest = MicrodatasusIngestManifest(
            status=status,
            request=request,
            request_hash=request.request_hash(),
            rscript_path=rscript_path,
            r_script_path=str(r_script_path),
            command=command,
            cwd=str(root),
            created_at=utc_now_iso(),
            started_at=started_at,
            completed_at=utc_now_iso(),
            elapsed_seconds=0.0,
            exit_code=0 if processed_exists else 31,
            paths=paths,
            r_manifest=load_json_if_exists(paths.r_manifest),
            raw_rows=raw_df.height,
            raw_cols=raw_df.width,
            processed_rows=None if processed_df is None else processed_df.height,
            processed_cols=None if processed_df is None else processed_df.width,
            raw_columns=raw_df.columns,
            processed_columns=[] if processed_df is None else processed_df.columns,
            raw_sha256=sha256_file(paths.raw_parquet),
            processed_sha256=(
                sha256_file(paths.processed_parquet)
                if processed_exists
                else None
            ),
            warnings=[
                "microdatasus_cache_hit",
                *([] if processed_exists else ["processed_cache_missing_raw_only"]),
            ],
        )
        write_json(paths.final_manifest_json, manifest)
        return manifest

    if not use_cache:
        prepare_microdatasus_work_dir_for_run(paths)
    
    write_json(
        paths.python_invocation,
        {
            "command": command,
            "cwd": str(root),
            "created_at": utc_now_iso(),
            "request": request.model_dump(mode="json"),
        },
    )

    try:
        exit_code = run_r_microdatasus(
            command=command,
            cwd=root,
            stdout_path=paths.stdout_log,
            stderr_path=paths.stderr_log,
            heartbeat_path=paths.heartbeat,
            timeout_seconds=timeout_seconds,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        )
    except FileNotFoundError as exc:
        manifest = MicrodatasusIngestManifest(
            status="error",
            request=request,
            request_hash=request.request_hash(),
            rscript_path=rscript_path,
            r_script_path=str(r_script_path),
            command=command,
            cwd=str(root),
            created_at=utc_now_iso(),
            started_at=started_at,
            completed_at=utc_now_iso(),
            elapsed_seconds=round(time.perf_counter() - start, 6),
            exit_code=None,
            paths=paths,
            error_message=f"Rscript not found: {exc}",
            warnings=["Rscript_not_found_on_PATH"],
        )
        write_json(paths.final_manifest_json, manifest)
        return manifest

    r_manifest = load_json_if_exists(paths.r_manifest)
    completed_at = utc_now_iso()
    elapsed = round(time.perf_counter() - start, 6)

    raw_csv_exists = Path(paths.raw_csv).exists()
    processed_csv_exists = Path(paths.processed_csv).exists()

    if exit_code != 0 and raw_csv_exists and not processed_csv_exists:
        raw_df = materialize_raw_only(request=request, paths=paths)
        error = (
            r_manifest_error_message(r_manifest)
            or stderr_tail(paths.stderr_log)
            or f"R microdatasus invocation failed with exit code {exit_code} after raw write."
        )

        manifest = MicrodatasusIngestManifest(
            status="raw_ok_process_error",
            request=request,
            request_hash=request.request_hash(),
            rscript_path=rscript_path,
            r_script_path=str(r_script_path),
            command=command,
            cwd=str(root),
            created_at=utc_now_iso(),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            paths=paths,
            r_manifest=r_manifest,
            error_message=error,
            raw_rows=raw_df.height,
            raw_cols=raw_df.width,
            processed_rows=None,
            processed_cols=None,
            raw_columns=raw_df.columns,
            processed_columns=[],
            raw_sha256=sha256_file(paths.raw_parquet),
            processed_sha256=None,
            stdout_sha256=sha256_file(paths.stdout_log)
            if Path(paths.stdout_log).exists()
            else None,
            stderr_sha256=sha256_file(paths.stderr_log)
            if Path(paths.stderr_log).exists()
            else None,
            warnings=[
                "microdatasus_process_failed_after_raw_fetch",
                "raw_parquet_materialized",
                "processed_parquet_missing",
            ],
        )
        write_json(paths.final_manifest_json, manifest)
        return manifest

    if exit_code != 0:
        error = (
            r_manifest_error_message(r_manifest)
            or stderr_tail(paths.stderr_log)
            or f"R microdatasus invocation failed with exit code {exit_code}."
        )

        manifest = MicrodatasusIngestManifest(
            status="error",
            request=request,
            request_hash=request.request_hash(),
            rscript_path=rscript_path,
            r_script_path=str(r_script_path),
            command=command,
            cwd=str(root),
            created_at=utc_now_iso(),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            paths=paths,
            r_manifest=r_manifest,
            error_message=error,
            stdout_sha256=sha256_file(paths.stdout_log)
            if Path(paths.stdout_log).exists()
            else None,
            stderr_sha256=sha256_file(paths.stderr_log)
            if Path(paths.stderr_log).exists()
            else None,
            warnings=["microdatasus_r_invocation_failed"],
        )
        write_json(paths.final_manifest_json, manifest)
        return manifest

    if not raw_csv_exists or not processed_csv_exists:
        manifest = MicrodatasusIngestManifest(
            status="error",
            request=request,
            request_hash=request.request_hash(),
            rscript_path=rscript_path,
            r_script_path=str(r_script_path),
            command=command,
            cwd=str(root),
            created_at=utc_now_iso(),
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            exit_code=exit_code,
            paths=paths,
            r_manifest=r_manifest,
            error_message="R exited successfully but raw.csv or processed.csv is missing.",
            warnings=["r_success_without_required_artifacts"],
        )
        write_json(paths.final_manifest_json, manifest)
        return manifest

    raw_df, processed_df = materialize_raw_and_processed(request=request, paths=paths)

    manifest = MicrodatasusIngestManifest(
        status="ok",
        request=request,
        request_hash=request.request_hash(),
        rscript_path=rscript_path,
        r_script_path=str(r_script_path),
        command=command,
        cwd=str(root),
        created_at=utc_now_iso(),
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
        exit_code=exit_code,
        paths=paths,
        r_manifest=r_manifest,
        raw_rows=raw_df.height,
        raw_cols=raw_df.width,
        processed_rows=processed_df.height,
        processed_cols=processed_df.width,
        raw_columns=raw_df.columns,
        processed_columns=processed_df.columns,
        raw_sha256=sha256_file(paths.raw_parquet),
        processed_sha256=sha256_file(paths.processed_parquet),
        stdout_sha256=sha256_file(paths.stdout_log)
        if Path(paths.stdout_log).exists()
        else None,
        stderr_sha256=sha256_file(paths.stderr_log)
        if Path(paths.stderr_log).exists()
        else None,
    )

    write_json(paths.final_manifest_json, manifest)
    return manifest