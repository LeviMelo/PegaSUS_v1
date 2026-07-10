from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pegasus.core.config import load_yaml
from pegasus.core.hashing import content_hash
from pegasus.core.schemas import DATASUSRequestManifest


ALLOWED_SYSTEMS = {"SIM-DO", "SINASC", "SIH-RD", "CNES-ST"}
UF_RE = re.compile(r"^[A-Z]{2}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_system(system: str) -> str:
    value = system.strip().upper()
    aliases = {
        "SIM": "SIM-DO",
        "SIM_DO": "SIM-DO",
        "SIM-DO": "SIM-DO",
        "DO": "SIM-DO",
        "SINASC": "SINASC",
        "SIH": "SIH-RD",
        "SIH_RD": "SIH-RD",
        "SIH-RD": "SIH-RD",
        "RD": "SIH-RD",
        "CNES": "CNES-ST",
        "CNES_ST": "CNES-ST",
        "CNES-ST": "CNES-ST",
        "ST": "CNES-ST",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported DATASUS system: {system}")
    return aliases[value]


def normalize_uf(uf: str) -> str:
    value = uf.strip().upper()
    if not UF_RE.fullmatch(value):
        raise ValueError(f"Invalid UF code: {uf}")
    return value


def parse_years(years: str) -> list[int]:
    result: set[int] = set()

    for part in years.split(","):
        token = part.strip()
        if not token:
            continue

        if "-" in token:
            left, right = [x.strip() for x in token.split("-", 1)]
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid descending year range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))

    if not result:
        raise ValueError("No years were parsed.")

    for year in result:
        if year < 1970 or year > 2100:
            raise ValueError(f"Suspicious DATASUS year: {year}")

    return sorted(result)


def period_label(
    *,
    year_start: int,
    year_end: int,
    month_start: int | None = None,
    month_end: int | None = None,
) -> str:
    ms = "NA" if month_start is None else f"{month_start:02d}"
    me = "NA" if month_end is None else f"{month_end:02d}"
    return f"{year_start}_{ms}__{year_end}_{me}"


def load_datasus_config(path: str | Path = "config/datasus.yaml") -> dict[str, Any]:
    data = load_yaml(path)
    return data.get("datasus", data)


@lru_cache(maxsize=4)
def _availability_windows(path: str = "config/datasus.yaml") -> dict[str, tuple[int, int]]:
    """Per-system earliest ``(first_year, first_month)`` a system exists in DATASUS.

    Read from ``config/datasus.yaml`` ``availability`` (see that file). Missing/malformed
    entries yield no floor for that system (fetch everything requested). Cached because the
    planner asks per (system, uf) across a whole national run.
    """
    try:
        cfg = load_datasus_config(path)
    except (OSError, ValueError, TypeError):
        return {}
    out: dict[str, tuple[int, int]] = {}
    for system, window in (cfg.get("availability") or {}).items():
        try:
            key = normalize_system(system)
        except ValueError:
            continue
        if isinstance(window, dict):
            first_year = int(window.get("first_year", 0))
            first_month = int(window.get("first_month", 1))
        else:
            first_year, first_month = int(window), 1
        out[key] = (first_year, min(12, max(1, first_month)))
    return out


@lru_cache(maxsize=4)
def _availability_ceilings(path: str = "config/datasus.yaml") -> dict[str, int]:
    """Per-system latest published ``last_year`` (publication lag). A system whose final data ends
    before the requested window's end (e.g. SINASC final = 2022) must not be requested past its
    ceiling, or a strict full-data run fails on a not-yet-published year. Additive to the first-year
    floor; systems without a ``last_year`` have no ceiling (assume current)."""
    try:
        cfg = load_datasus_config(path)
    except (OSError, ValueError, TypeError):
        return {}
    out: dict[str, int] = {}
    for system, window in (cfg.get("availability") or {}).items():
        if not isinstance(window, dict) or "last_year" not in window:
            continue
        try:
            out[normalize_system(system)] = int(window["last_year"])
        except (ValueError, TypeError):
            continue
    return out


@lru_cache(maxsize=4)
def _known_unavailable_periods(
    path: str = "config/datasus.yaml",
) -> frozenset[tuple[str, str, int, int | None]]:
    """Evidence-backed holes within a system's general availability window."""
    try:
        cfg = load_datasus_config(path)
    except (OSError, ValueError, TypeError):
        return frozenset()
    out: set[tuple[str, str, int, int | None]] = set()
    for raw_system, entries in (cfg.get("known_unavailable") or {}).items():
        try:
            system = normalize_system(raw_system)
        except ValueError:
            continue
        for entry in entries or ():
            if not isinstance(entry, dict):
                continue
            try:
                uf = normalize_uf(str(entry["uf"]))
                year = int(entry["year"])
            except (KeyError, TypeError, ValueError):
                continue
            months = entry.get("months")
            if months is None:
                out.add((system, uf, year, None))
            else:
                for month in months:
                    month_int = int(month)
                    if 1 <= month_int <= 12:
                        out.add((system, uf, year, month_int))
    return frozenset(out)


def _is_available(system: str, year: int, month: int | None, *, uf: str | None = None) -> bool:
    """True if DATASUS publishes ``system`` at ``(year, month)`` per the availability floor.

    A chunk below the floor is not emitted — DATASUS has no file there, so fetching it only
    spins up a doomed R subprocess. For annual systems (``month is None``) only the year floor
    applies. Systems without a configured floor are always available (no filtering)."""
    system = normalize_system(system)
    floor = _availability_windows().get(system)
    if floor is None:
        return True
    first_year, first_month = floor
    if year < first_year:
        return False
    if year == first_year and month is not None and month < first_month:
        return False
    ceiling = _availability_ceilings().get(system)
    if ceiling is not None and year > ceiling:
        return False
    if uf is not None and (system, normalize_uf(uf), year, month) in _known_unavailable_periods():
        return False
    return True


def _request_identity(
    *,
    system: str,
    uf: str,
    year_start: int,
    year_end: int,
    month_start: int | None,
    month_end: int | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "backend": config.get("backend", "microdatasus"),
        "system": system,
        "uf": uf,
        "year_start": year_start,
        "year_end": year_end,
        "month_start": month_start,
        "month_end": month_end,
        "information_system": system,
        "fetch_function": "fetch_datasus",
        "process_function": "pegasus_process_datasus_dispatch",
        "processing_contract_version": "datasus_r_bridge_v2_raw_canonical_plus_microdatasus_sidecar",
    }


def build_datasus_request_manifest(
    *,
    system: str,
    uf: str,
    year_start: int,
    year_end: int,
    month_start: int | None = None,
    month_end: int | None = None,
    config: dict[str, Any] | None = None,
    data_root: str | Path = "data",
) -> DATASUSRequestManifest:
    config = config or load_datasus_config()
    system = normalize_system(system)
    uf = normalize_uf(uf)

    identity = _request_identity(
        system=system,
        uf=uf,
        year_start=year_start,
        year_end=year_end,
        month_start=month_start,
        month_end=month_end,
        config=config,
    )
    request_hash = content_hash(identity)
    period = period_label(
        year_start=year_start,
        year_end=year_end,
        month_start=month_start,
        month_end=month_end,
    )

    data_root = Path(data_root)

    raw_dir = data_root / "raw" / "datasus" / system / f"uf={uf}" / f"period={period}" / request_hash
    processed_dir = data_root / "processed" / "datasus" / system / f"uf={uf}" / f"period={period}" / request_hash

    now = utc_now()

    return DATASUSRequestManifest(
        system=system,
        uf=uf,
        year_start=year_start,
        month_start=month_start,
        year_end=year_end,
        month_end=month_end,
        information_system=identity["information_system"],
        fetch_function=identity["fetch_function"],
        process_function=identity["process_function"],
        raw_path=str(raw_dir / "raw.rds"),
        processed_path=str(processed_dir / "processed.parquet"),
        raw_sha256="",
        processed_sha256="",
        row_counts={},
        column_lists={},
        started_at=now,
        ended_at=now,
        duration_seconds=0.0,
        rscript_path=str(config.get("rscript_path", "Rscript")),
        r_version=None,
        microdatasus_version=None,
        read_dbc_version=None,
        stdout_path=str(raw_dir / "stdout.log"),
        stderr_path=str(raw_dir / "stderr.log"),
        heartbeat_path=str(raw_dir / "heartbeat.json"),
        exit_code=41,
        status="blocked",
        error_message="not_executed",
        request_hash=request_hash,
    )


def build_datasus_manifests(
    *,
    system: str,
    uf: str,
    years: str,
    config: dict[str, Any] | None = None,
    data_root: str | Path = "data",
) -> list[DATASUSRequestManifest]:
    config = config or load_datasus_config()
    system = normalize_system(system)
    uf = normalize_uf(uf)
    parsed_years = parse_years(years)

    # SIM-DO and SINASC are annual UF files. SIH-RD and CNES-ST are monthly
    # DATASUS systems. Default is PER-MONTH chunks: microdatasus's `process_*`
    # canonical step is super-linear, so a per-year chunk (~170k SIH rows) makes
    # one R process hang for many minutes — and 16 such heavy processes contend
    # for CPU. Per-month keeps each R process light (~13k rows, ~30s) so throughput
    # comes from download parallelism (many workers) instead of few huge chunks.
    # Set PEGASUS_DATASUS_YEAR_CHUNKS=1 to batch per-year (fewer R startups, but
    # slow/contended processing — not recommended for monthly systems).
    if system in {"SIH-RD", "CNES-ST"}:
        import os

        if os.environ.get("PEGASUS_DATASUS_YEAR_CHUNKS", "0") == "1":
            return [
                build_datasus_request_manifest(
                    system=system, uf=uf, year_start=year, year_end=year,
                    month_start=1, month_end=12, config=config, data_root=data_root,
                )
                for year in parsed_years
                # Keep any year that has ≥1 available month (partial availability years
                # still fetch the whole year; microdatasus returns what exists).
                if any(_is_available(system, year, month, uf=uf) for month in range(1, 13))
            ]
        return [
            build_datasus_request_manifest(
                system=system, uf=uf, year_start=year, year_end=year,
                month_start=month, month_end=month, config=config, data_root=data_root,
            )
            for year in parsed_years
            for month in range(1, 13)
            if _is_available(system, year, month, uf=uf)
        ]

    return [
        build_datasus_request_manifest(
            system=system,
            uf=uf,
            year_start=year,
            year_end=year,
            month_start=None,
            month_end=None,
            config=config,
            data_root=data_root,
        )
        for year in parsed_years
        if _is_available(system, year, None, uf=uf)
    ]


def manifest_output_path(
    manifest: DATASUSRequestManifest,
    *,
    root: str | Path = "data/manifests/datasus",
) -> Path:
    period = period_label(
        year_start=manifest.year_start,
        year_end=manifest.year_end,
        month_start=manifest.month_start,
        month_end=manifest.month_end,
    )
    return (
        Path(root)
        / manifest.system
        / f"uf={manifest.uf}"
        / f"period={period}"
        / manifest.request_hash
        / "manifest.json"
    )


def write_request_manifest(
    manifest: DATASUSRequestManifest,
    *,
    root: str | Path = "data/manifests/datasus",
) -> Path:
    path = manifest_output_path(manifest, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def read_request_manifest(path: str | Path) -> DATASUSRequestManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DATASUSRequestManifest.model_validate(payload)
