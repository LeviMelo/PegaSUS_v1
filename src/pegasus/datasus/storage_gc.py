"""DATASUS storage garbage collection — reclaim redundant per-chunk sidecars.

Each DATASUS chunk historically wrote three serializations of the same rows: ``raw.rds``
(R-native, never read back), ``microdatasus_processed.parquet`` (the microdatasus semantic
copy, not consumed by the in-house codebook normalizer), and ``processed.parquet`` (the one
the pipeline consumes). This GC deletes the first two plus the debug ancillaries (stdout/
stderr/heartbeat), keeping ``manifest.json`` (cache + provenance) and ``processed.parquet``.

It is **safe by construction**: a chunk's sidecars are removed only after the corresponding
``processed.parquet`` is confirmed present, so nothing that PegaSUS reads is ever deleted, and
no re-fetch is triggered (the cache is keyed on ``processed.parquet`` + ``manifest.json``).
``dry_run=True`` by default — it reports what it would reclaim without touching disk.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

# Redundant per-chunk artifacts removed by the GC. `manifest.json` and `processed.parquet` are
# never in this set.
_REDUNDANT_SIDECARS = ("raw.rds", "microdatasus_processed.parquet")
_DEBUG_ANCILLARIES = ("stdout.log", "stderr.log", "heartbeat.json")

# Per-run stage workspaces that accrete as suffixed siblings of a run bundle. Both are regenerable
# stage scratch, never a canonical bundle:
#   * ``__pirs_stage_workspace`` — written by the removed PIRS stage; ZERO live writers remain, so
#     pure dead-path detritus (superseded by the sibling ``run/`` bundle).
#   * ``__efg_stage_workspace`` — a transient full-duplicate of the tensor payload; the live compile
#     path already deletes it after a successful flush (workflows/compile.py, T1.4), so any on-disk
#     copy is a leftover from a pre-T1.4 or interrupted run.
_STALE_WORKSPACE_SUFFIXES = ("__pirs_stage_workspace", "__efg_stage_workspace")


@dataclass
class GCStats:
    chunks_scanned: int = 0
    chunks_reclaimed: int = 0
    chunks_skipped_no_processed: int = 0
    bytes_reclaimed: int = 0
    files_deleted: int = 0
    removed_by_name: dict[str, int] = field(default_factory=dict)
    skipped_chunks: list[str] = field(default_factory=list)

    def note_removed(self, name: str, size: int) -> None:
        self.files_deleted += 1
        self.bytes_reclaimed += size
        self.removed_by_name[name] = self.removed_by_name.get(name, 0) + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "chunks_scanned": self.chunks_scanned,
            "chunks_reclaimed": self.chunks_reclaimed,
            "chunks_skipped_no_processed": self.chunks_skipped_no_processed,
            "files_deleted": self.files_deleted,
            "bytes_reclaimed": self.bytes_reclaimed,
            "gib_reclaimed": round(self.bytes_reclaimed / 1024**3, 2),
            "removed_by_name": dict(sorted(self.removed_by_name.items())),
            "skipped_chunk_sample": self.skipped_chunks[:10],
        }


def _processed_parquet_for(raw_chunk_dir: Path, data_root: Path) -> Path:
    """Map a raw chunk dir (``…/raw/datasus/SYS/uf=X/period=Y/hash``) to its consumed
    ``processed.parquet`` under ``…/processed/datasus/…/hash/processed.parquet``."""
    rel = raw_chunk_dir.relative_to(data_root / "raw" / "datasus")
    return data_root / "processed" / "datasus" / rel / "processed.parquet"


def _raw_chunk_dirs(data_root: Path):
    """Yield chunk dirs under ``data/raw/datasus`` — those containing a ``manifest.json``
    (the R sidecar), which marks a real acquisition chunk rather than a partition level."""
    root = data_root / "raw" / "datasus"
    if not root.exists():
        return
    for manifest in root.rglob("manifest.json"):
        yield manifest.parent


def gc_datasus_raw_sidecars(
    *, data_root: str | Path = "data", dry_run: bool = True, prune_debug: bool = True
) -> GCStats:
    """Delete redundant ``raw.rds`` + ``microdatasus_processed.parquet`` (and, if
    ``prune_debug``, stdout/stderr/heartbeat) from every DATASUS chunk whose
    ``processed.parquet`` exists. Keeps ``manifest.json``. Returns a :class:`GCStats`."""
    data_root = Path(data_root)
    stats = GCStats()
    targets = list(_REDUNDANT_SIDECARS) + (list(_DEBUG_ANCILLARIES) if prune_debug else [])

    for chunk_dir in _raw_chunk_dirs(data_root):
        stats.chunks_scanned += 1
        processed = _processed_parquet_for(chunk_dir, data_root)
        if not processed.exists():
            # Never strip a chunk we cannot re-derive from — leave it intact for re-fetch.
            stats.chunks_skipped_no_processed += 1
            stats.skipped_chunks.append(str(chunk_dir))
            continue

        reclaimed_here = False
        for name in targets:
            target = chunk_dir / name
            if not target.exists():
                continue
            size = target.stat().st_size
            if not dry_run:
                try:
                    target.unlink()
                except OSError:
                    continue
            stats.note_removed(name, size)
            reclaimed_here = True
        if reclaimed_here:
            stats.chunks_reclaimed += 1

    return stats


def gc_sidra_raw_payloads(
    *, data_root: str | Path = "data", dry_run: bool = True
) -> GCStats:
    """Strip the duplicated response ``payload`` from existing SIDRA extract dumps
    (``data/sidra/**/raw/*.json``), rewriting them compact.

    The payload is already archived in the SidraClient cache (``data/cache/sidra``), so the
    inline copy in each extract dump is redundant. Successful chunks are slimmed to a provenance
    record (request + sidecar + payload hash); failure dumps keep their (small) error payload for
    debugging. The file is retained so provenance references stay valid — only the bytes shrink.
    """
    data_root = Path(data_root)
    stats = GCStats()
    sidra_root = data_root / "sidra"
    if not sidra_root.exists():
        return stats

    for dump in sidra_root.rglob("raw/*.json"):
        stats.chunks_scanned += 1
        try:
            before = dump.stat().st_size
            record = json.loads(dump.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or "payload" not in record:
            continue
        if int(record.get("status_code", 200)) >= 400:
            # keep small error payloads inline
            continue
        record.pop("payload", None)
        record.setdefault("payload_archive", "data/cache/sidra (namespace=values)")
        if not dry_run:
            try:
                dump.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            except OSError:
                continue
        after = len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        saved = max(0, before - after)
        stats.chunks_reclaimed += 1
        stats.note_removed("sidra_payload_bytes", saved)

    return stats


def _dir_size_and_count(path: Path) -> tuple[int, int]:
    total = 0
    n = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
                n += 1
        except OSError:
            continue
    return total, n


def _workspace_age_seconds(path: Path, now: float) -> float:
    """Age of the newest entry (dir or its top-level children) — so an actively-written workspace
    reads as young and is skipped."""
    newest = path.stat().st_mtime
    try:
        for child in path.iterdir():
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return now - newest


def gc_by_contract(
    *, repo_root: str | Path = ".", dry_run: bool = True, now: float | None = None
) -> GCStats:
    """Reclaim reclaimable data per the persistence contract (:mod:`pegasus.core.data_lifecycle`).

    Honors exactly two autonomous policies — CACHE roots are dropped file-by-file once older than
    their ``ttl_days``, and EPHEMERAL (DELETE_ALWAYS) roots have their contents removed. Everything
    else (canonical inputs, versioned assets, durable run outputs, metadata, test fixtures) is
    NEVER touched here — those are user-gated. ``dry_run=True`` by default. ``now`` is injectable
    for tests.
    """
    from pegasus.core.data_lifecycle import GCPolicy, roots_with_policy

    repo_root = Path(repo_root)
    stats = GCStats()
    now = time.time() if now is None else now

    # CACHE: drop files older than the root's TTL.
    for root in roots_with_policy(GCPolicy.DROP_BY_MTIME):
        base = repo_root / root.rel
        if not base.exists() or root.ttl_days is None:
            continue
        cutoff = now - root.ttl_days * 86400.0
        for path in base.rglob("*"):
            try:
                if not path.is_file():
                    continue
                st = path.stat()
                if st.st_mtime >= cutoff:
                    continue
                size = st.st_size
            except OSError:
                continue
            if not dry_run:
                try:
                    path.unlink()
                except OSError:
                    continue
            stats.note_removed(f"cache_expired:{root.rel}", size)
            stats.chunks_reclaimed += 1
        stats.chunks_scanned += 1

    # EPHEMERAL: remove the contents of always-delete roots (keep the dir + .gitkeep).
    for root in roots_with_policy(GCPolicy.DELETE_ALWAYS):
        base = repo_root / root.rel
        if not base.exists():
            continue
        stats.chunks_scanned += 1
        for child in base.rglob("*"):
            try:
                if not child.is_file() or child.name == ".gitkeep":
                    continue
                size = child.stat().st_size
            except OSError:
                continue
            if not dry_run:
                try:
                    child.unlink()
                except OSError:
                    continue
            stats.note_removed(f"ephemeral:{root.rel}", size)
            stats.chunks_reclaimed += 1

    return stats


def gc_stale_stage_workspaces(
    *, data_root: str | Path = "data", dry_run: bool = True, min_age_seconds: float = 3600.0
) -> GCStats:
    """Reclaim orphaned per-run stage workspaces (``*__pirs_stage_workspace`` /
    ``*__efg_stage_workspace``).

    **Safe by construction**: it matches only the workspace *suffix*, so a canonical ``run/`` or
    ``runs/<id>/`` bundle is never a candidate. Workspaces whose newest entry is younger than
    ``min_age_seconds`` are skipped, so a concurrently-running compile is never raced. ``dry_run=True``
    by default — it reports what it would reclaim without touching disk. Returns a :class:`GCStats`
    (``chunks_scanned`` = workspaces seen, ``chunks_reclaimed`` = workspaces removed,
    ``removed_by_name`` keyed by suffix).
    """
    data_root = Path(data_root)
    stats = GCStats()
    if not data_root.exists():
        return stats
    now = time.time()

    for suffix in _STALE_WORKSPACE_SUFFIXES:
        for path in data_root.rglob(f"*{suffix}"):
            if not path.is_dir():
                continue
            stats.chunks_scanned += 1
            if _workspace_age_seconds(path, now) < min_age_seconds:
                stats.chunks_skipped_no_processed += 1
                stats.skipped_chunks.append(str(path))
                continue
            size, n_files = _dir_size_and_count(path)
            if not dry_run:
                try:
                    shutil.rmtree(path)
                except OSError:
                    continue
            stats.files_deleted += n_files
            stats.bytes_reclaimed += size
            stats.removed_by_name[suffix] = stats.removed_by_name.get(suffix, 0) + 1
            stats.chunks_reclaimed += 1

    return stats


__all__ = [
    "GCStats",
    "gc_datasus_raw_sidecars",
    "gc_sidra_raw_payloads",
    "gc_stale_stage_workspaces",
    "gc_by_contract",
]
