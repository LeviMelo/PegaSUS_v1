"""The data-lifecycle persistence contract and its contract-driven GC (single source of truth)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from pegasus.core.data_lifecycle import (
    GCPolicy,
    Role,
    classify,
    is_safe_to_delete,
)
from pegasus.core.paths import DATA_LAKE_DIRS, validate_lake_against_contract
from pegasus.datasus.storage_gc import gc_by_contract


def test_classify_picks_longest_prefix():
    # a more specific root overrides its parent
    assert classify("data/processed/sidra/facts/population_2022.parquet").rel == "data/processed/sidra/facts"
    assert classify("data/processed/datasus/SIM-DO/x/processed.parquet").rel == "data/processed"
    assert classify("data/cache/sidra/http/abc.json").role is Role.CACHE
    assert classify("data/runs/compile_x/V_fields.parquet").role is Role.DURABLE_OUTPUT
    assert classify("data/raw/datasus/SIM-DO/x").role is Role.CANONICAL_INPUT
    assert classify("data/assets/population_tensor/v2025.1").role is Role.ASSET
    assert classify("nowhere/at/all") is None


def test_is_safe_to_delete_only_cache_and_ephemeral():
    assert is_safe_to_delete("data/cache/sidra/http/x.json")          # cache: yes
    assert is_safe_to_delete("data/intermediate/she/scratch")         # ephemeral: yes
    assert not is_safe_to_delete("data/raw/datasus/SIM-DO/x")         # canonical input: never
    assert not is_safe_to_delete("data/runs/compile_x")               # durable output: user-gated
    assert not is_safe_to_delete("data/assets/population_tensor/v1")  # asset: user-gated
    assert not is_safe_to_delete("data/manifests/runs/x.json")        # metadata: never


def test_lake_skeleton_agrees_with_contract():
    # every pre-created dir must fall under a governed lifecycle root (no drift)
    assert validate_lake_against_contract() == []
    # and none of the pre-created dirs is an ephemeral/test-fixture root
    for rel in DATA_LAKE_DIRS:
        root = classify(rel)
        assert root is not None
        assert root.role not in (Role.EPHEMERAL, Role.TEST_FIXTURE)


def _touch(path: Path, age_days: float, now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 100)
    t = now - age_days * 86400.0
    os.utime(path, (t, t))


def test_gc_by_contract_drops_old_cache_keeps_fresh_and_canonical(tmp_path):
    now = time.time()
    old_cache = tmp_path / "data/cache/sidra/http/old.json"
    fresh_cache = tmp_path / "data/cache/sidra/http/fresh.json"
    canonical = tmp_path / "data/raw/datasus/SIM-DO/old_raw.parquet"
    ephemeral = tmp_path / "data/intermediate/she/scratch.parquet"
    _touch(old_cache, age_days=40, now=now)     # older than 30d cache TTL
    _touch(fresh_cache, age_days=1, now=now)    # within TTL
    _touch(canonical, age_days=999, now=now)    # ancient but NEVER GC
    _touch(ephemeral, age_days=0.1, now=now)    # ephemeral: delete regardless of age

    stats = gc_by_contract(repo_root=tmp_path, dry_run=False, now=now)

    assert not old_cache.exists()               # expired cache reclaimed
    assert fresh_cache.exists()                 # in-TTL cache kept
    assert canonical.exists()                   # canonical input never touched
    assert not ephemeral.exists()               # ephemeral reclaimed
    assert stats.bytes_reclaimed > 0


def test_gc_by_contract_dry_run_touches_nothing(tmp_path):
    now = time.time()
    old_cache = tmp_path / "data/cache/datasus/microdatasus/old.json"
    _touch(old_cache, age_days=40, now=now)

    stats = gc_by_contract(repo_root=tmp_path, dry_run=True, now=now)

    assert stats.bytes_reclaimed > 0            # counted
    assert old_cache.exists()                   # but not performed
