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

from dataclasses import dataclass, field
from pathlib import Path

# Redundant per-chunk artifacts removed by the GC. `manifest.json` and `processed.parquet` are
# never in this set.
_REDUNDANT_SIDECARS = ("raw.rds", "microdatasus_processed.parquet")
_DEBUG_ANCILLARIES = ("stdout.log", "stderr.log", "heartbeat.json")


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


__all__ = ["GCStats", "gc_datasus_raw_sidecars"]
