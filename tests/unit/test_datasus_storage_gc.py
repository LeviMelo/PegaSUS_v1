"""DATASUS storage GC: reclaims redundant sidecars only when the consumed artifact is safe."""

from __future__ import annotations

import os
import time
from pathlib import Path

import json

from pegasus.datasus.storage_gc import (
    gc_datasus_raw_sidecars,
    gc_sidra_raw_payloads,
    gc_stale_stage_workspaces,
)


def _make_chunk(data_root: Path, *, hash_id: str, with_processed: bool) -> Path:
    raw = data_root / "raw" / "datasus" / "SIM-DO" / "uf=AL" / "period=2022_NA__2022_NA" / hash_id
    raw.mkdir(parents=True)
    (raw / "manifest.json").write_text('{"status":"success"}', encoding="utf-8")
    (raw / "raw.rds").write_bytes(b"x" * 1000)
    (raw / "microdatasus_processed.parquet").write_bytes(b"y" * 500)
    (raw / "stdout.log").write_text("", encoding="utf-8")
    (raw / "stderr.log").write_text("boom", encoding="utf-8")
    (raw / "heartbeat.json").write_text("{}", encoding="utf-8")
    if with_processed:
        proc = data_root / "processed" / "datasus" / "SIM-DO" / "uf=AL" / "period=2022_NA__2022_NA" / hash_id
        proc.mkdir(parents=True)
        (proc / "processed.parquet").write_bytes(b"z" * 2000)
    return raw


def test_gc_reclaims_sidecars_keeps_manifest(tmp_path: Path) -> None:
    raw = _make_chunk(tmp_path, hash_id="h1", with_processed=True)

    stats = gc_datasus_raw_sidecars(data_root=tmp_path, dry_run=False, prune_debug=True)

    assert stats.chunks_reclaimed == 1
    assert stats.chunks_skipped_no_processed == 0
    # Redundant sidecars + debug ancillaries gone
    assert not (raw / "raw.rds").exists()
    assert not (raw / "microdatasus_processed.parquet").exists()
    assert not (raw / "stderr.log").exists()
    assert not (raw / "heartbeat.json").exists()
    # Provenance + consumed artifact preserved
    assert (raw / "manifest.json").exists()
    proc = tmp_path / "processed" / "datasus" / "SIM-DO" / "uf=AL" / "period=2022_NA__2022_NA" / "h1" / "processed.parquet"
    assert proc.exists()


def test_gc_skips_chunk_without_processed(tmp_path: Path) -> None:
    # A chunk whose processed.parquet is absent must be left fully intact (re-fetchable).
    raw = _make_chunk(tmp_path, hash_id="h2", with_processed=False)

    stats = gc_datasus_raw_sidecars(data_root=tmp_path, dry_run=False, prune_debug=True)

    assert stats.chunks_reclaimed == 0
    assert stats.chunks_skipped_no_processed == 1
    assert (raw / "raw.rds").exists()
    assert (raw / "microdatasus_processed.parquet").exists()


def test_gc_dry_run_touches_nothing(tmp_path: Path) -> None:
    raw = _make_chunk(tmp_path, hash_id="h3", with_processed=True)

    stats = gc_datasus_raw_sidecars(data_root=tmp_path, dry_run=True, prune_debug=True)

    assert stats.bytes_reclaimed > 0
    assert stats.files_deleted > 0  # counted, not performed
    assert (raw / "raw.rds").exists()  # dry-run left disk untouched
    assert (raw / "microdatasus_processed.parquet").exists()


def _sidra_dump(tmp_path: Path, *, name: str, status: int) -> Path:
    d = tmp_path / "sidra" / "population_totals_AL_9606" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(
        json.dumps({
            "chunk": {"chunk_id": name},
            "status_code": status,
            "from_cache": False,
            "sidecar": {"sha256": "abc", "bytes": 999},
            "payload": [{"big": "x" * 5000}],
        }),
        encoding="utf-8",
    )
    return p


def test_sidra_gc_strips_success_payload_keeps_failure(tmp_path: Path) -> None:
    ok = _sidra_dump(tmp_path, name="ok", status=200)
    bad = _sidra_dump(tmp_path, name="bad", status=599)
    before_ok = ok.stat().st_size

    stats = gc_sidra_raw_payloads(data_root=tmp_path, dry_run=False)

    assert stats.chunks_reclaimed == 1
    assert ok.stat().st_size < before_ok
    ok_rec = json.loads(ok.read_text(encoding="utf-8"))
    assert "payload" not in ok_rec          # success payload stripped (archived in client cache)
    assert ok_rec["sidecar"]["sha256"] == "abc"  # provenance retained
    bad_rec = json.loads(bad.read_text(encoding="utf-8"))
    assert "payload" in bad_rec             # failure payload kept for debugging


def _make_run_with_workspaces(data_root: Path, *, age_seconds: float) -> tuple[Path, Path, Path]:
    runs = data_root / "runs"
    bundle = runs / "run_abc"
    bundle.mkdir(parents=True)
    (bundle / "Q_tensor.parquet").write_bytes(b"real" * 100)  # canonical output — never a candidate
    pirs_ws = runs / "run_abc__pirs_stage_workspace"
    pirs_ws.mkdir(parents=True)
    (pirs_ws / "Q_tensor.parquet").write_bytes(b"scratch" * 100)
    efg_ws = runs / "run_abc__efg_stage_workspace"
    efg_ws.mkdir(parents=True)
    (efg_ws / "V_fields.parquet").write_bytes(b"scratch" * 100)
    old = time.time() - age_seconds
    for ws in (pirs_ws, efg_ws):
        for p in ws.rglob("*"):
            os.utime(p, (old, old))
        os.utime(ws, (old, old))
    return bundle, pirs_ws, efg_ws


def test_workspace_gc_reclaims_scratch_keeps_bundle(tmp_path: Path) -> None:
    bundle, pirs_ws, efg_ws = _make_run_with_workspaces(tmp_path, age_seconds=10_000)

    stats = gc_stale_stage_workspaces(data_root=tmp_path, dry_run=False, min_age_seconds=3600)

    assert stats.chunks_reclaimed == 2
    assert not pirs_ws.exists()  # dead PIRS scratch reclaimed
    assert not efg_ws.exists()   # stale EFG duplicate reclaimed
    assert bundle.exists() and (bundle / "Q_tensor.parquet").exists()  # canonical bundle untouched
    assert stats.removed_by_name.get("__pirs_stage_workspace") == 1
    assert stats.removed_by_name.get("__efg_stage_workspace") == 1


def test_workspace_gc_dry_run_touches_nothing(tmp_path: Path) -> None:
    _, pirs_ws, efg_ws = _make_run_with_workspaces(tmp_path, age_seconds=10_000)

    stats = gc_stale_stage_workspaces(data_root=tmp_path, dry_run=True, min_age_seconds=3600)

    assert stats.chunks_reclaimed == 2
    assert stats.bytes_reclaimed > 0  # counted, not performed
    assert pirs_ws.exists() and efg_ws.exists()


def test_workspace_gc_skips_recent_workspace(tmp_path: Path) -> None:
    # A freshly-written workspace (age < min_age) is skipped so an in-flight compile is never raced.
    _, pirs_ws, efg_ws = _make_run_with_workspaces(tmp_path, age_seconds=5)

    stats = gc_stale_stage_workspaces(data_root=tmp_path, dry_run=False, min_age_seconds=3600)

    assert stats.chunks_reclaimed == 0
    assert stats.chunks_skipped_no_processed == 2
    assert pirs_ws.exists() and efg_ws.exists()
