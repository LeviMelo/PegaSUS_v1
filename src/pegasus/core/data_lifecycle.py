"""The PegaSUS data lifecycle + persistence contract — single source of truth.

Every path under ``data/`` has a ROLE (what kind of thing it is) and a GC POLICY (whether and how it
may be reclaimed). Historically this was convention scattered across comments; at national scale the
absence of a durable-vs-ephemeral law lets ``data/`` accrete without bound and makes "is this safe to
delete?" unanswerable. This module is that law in code:

- :func:`ensure_data_lake` creates the durable skeleton from it (skipping ephemeral/test roots);
- the storage GC (:mod:`pegasus.datasus.storage_gc`) honors the policies (drop cache by age, delete
  ephemeral, never touch canonical inputs / user-gated outputs);
- :func:`classify` lets any tool ask a path's role instead of guessing from the string.

The verified lifecycle (see the reconnaissance in PEGASUS_COMPLETION_ROADMAP.md §3):
    raw/datasus (FTP→R, raw cols) → processed/datasus (R output, raw cols) → normalized/datasus
    (SHE-canonical) ; SIDRA cache → processed/sidra/facts ; both → assets/population_tensor/v{y}.{s}
    (build-once, sliced on query) → runs/{run_id}/ (the 17-key bundle).
``data/runs`` is the ONLY production run root; ``actual_state_panels``/``actual_smokes`` are dev/test
snapshots (never written by ``src``).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Role(str, Enum):
    CANONICAL_INPUT = "canonical_input"   # externally-fetched inputs; re-fetch is costly → never auto-GC
    DERIVED = "derived"                    # rebuildable from canonical inputs → droppable (rebuild if missing)
    ASSET = "asset"                        # versioned, immutable, build-once foundational assets
    CACHE = "cache"                        # fetch/compute cache → drop by age
    DURABLE_OUTPUT = "durable_output"      # run bundles; retention is the user's call
    METADATA = "metadata"                  # manifests/provenance; kept with what they describe
    EPHEMERAL = "ephemeral"                # scratch/stale; safe to delete
    TEST_FIXTURE = "test_fixture"          # blessed dev/test snapshots; not written by src


class GCPolicy(str, Enum):
    NEVER = "never"                         # never reclaim (canonical inputs, provenance)
    REBUILD_IF_MISSING = "rebuild_if_missing"  # droppable; the pipeline rebuilds on demand
    DROP_BY_MTIME = "drop_by_mtime"        # reclaim entries older than ttl_days
    DELETE_ALWAYS = "delete_always"        # stale/unused; remove whenever seen
    USER_GATED = "user_gated"              # only the user may reclaim (durable outputs, versioned assets)


@dataclass(frozen=True)
class DataRoot:
    rel: str                      # repo-relative path, forward-slashed (e.g. "data/cache")
    role: Role
    gc_policy: GCPolicy
    ttl_days: int | None          # meaningful only for DROP_BY_MTIME
    rebuildable: bool
    description: str
    create: bool = True           # whether ensure_data_lake pre-creates it


# The contract. Order does not matter — classify() picks the longest matching prefix, so a more
# specific root (processed/sidra/facts) can override its parent (processed).
DATA_LIFECYCLE: tuple[DataRoot, ...] = (
    DataRoot("data/raw", Role.CANONICAL_INPUT, GCPolicy.NEVER, None, False,
             "Externally-fetched raw inputs (DATASUS FTP via the R bridge; raw geo/external). Re-fetch is costly."),
    DataRoot("data/processed/sidra/facts", Role.DERIVED, GCPolicy.REBUILD_IF_MISSING, None, True,
             "Canonical SIDRA population facts (9606 etc.). Rebuildable from the SIDRA cache."),
    DataRoot("data/processed", Role.DERIVED, GCPolicy.REBUILD_IF_MISSING, None, True,
             "R-bridge output (still raw DATASUS columns) + combine intermediates. Rebuildable from raw."),
    DataRoot("data/normalized", Role.DERIVED, GCPolicy.REBUILD_IF_MISSING, None, True,
             "SHE-canonical decoded DATASUS events (race_color_admin, parsed ICD). Rebuildable from processed."),
    DataRoot("data/assets", Role.ASSET, GCPolicy.USER_GATED, None, False,
             "Versioned build-once foundational assets (population_tensor/v{year}.{seq}), sliced on query."),
    DataRoot("data/cache", Role.CACHE, GCPolicy.DROP_BY_MTIME, 30, True,
             "DATASUS/SIDRA fetch + value caches. Reconstructable by re-fetch; drop by age."),
    DataRoot("data/metadata", Role.METADATA, GCPolicy.NEVER, None, True,
             "SIDRA/DATASUS metadata + normalized metadata tables + registry caches."),
    DataRoot("data/manifests", Role.METADATA, GCPolicy.NEVER, None, True,
             "Per-chunk fetch + per-run provenance manifests."),
    DataRoot("data/runs", Role.DURABLE_OUTPUT, GCPolicy.USER_GATED, None, True,
             "Canonical compile/investigate run bundles — the ONLY production run root."),
    DataRoot("data/diagnostics", Role.METADATA, GCPolicy.DROP_BY_MTIME, 90, True,
             "Per-run telemetry/heartbeat diagnostics."),
    DataRoot("data/sidra", Role.DERIVED, GCPolicy.REBUILD_IF_MISSING, None, False,
             "LEGACY per-UF population-strata / civil-registry artifacts, superseded by data/assets. Rebuildable."),
    DataRoot("data/intermediate", Role.EPHEMERAL, GCPolicy.DELETE_ALWAYS, None, False,
             "Unused stage-workspace lake dirs — real per-run workspaces are written adjacent to the run_dir."),
    DataRoot("data/actual_state_panels", Role.TEST_FIXTURE, GCPolicy.USER_GATED, None, False,
             "Dev/test state-panel run snapshots. NOT written by src (only .codex-tmp/.tmp scripts)."),
    DataRoot("data/actual_smokes", Role.TEST_FIXTURE, GCPolicy.USER_GATED, None, False,
             "Dev/test smoke run snapshots. NOT written by src."),
)


def _norm(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").rstrip("/")


def classify(path: str | Path) -> DataRoot | None:
    """Return the governing :class:`DataRoot` for a path (longest matching prefix), or None."""
    p = _norm(path)
    best: DataRoot | None = None
    for root in DATA_LIFECYCLE:
        rel = root.rel
        if p == rel or p.startswith(rel + "/"):
            if best is None or len(rel) > len(best.rel):
                best = root
    return best


def roots_with_policy(policy: GCPolicy) -> list[DataRoot]:
    return [r for r in DATA_LIFECYCLE if r.gc_policy == policy]


def is_safe_to_delete(path: str | Path) -> bool:
    """True only for roots the GC may reclaim autonomously (cache-by-age or always-delete).

    Canonical inputs, assets, durable outputs, metadata, and test fixtures return False — those are
    never reclaimed without an explicit user decision.
    """
    root = classify(path)
    return root is not None and root.gc_policy in (GCPolicy.DROP_BY_MTIME, GCPolicy.DELETE_ALWAYS)


def persistence_report() -> str:
    """A human-readable rendering of the contract (for docs / `pegasus` introspection)."""
    lines = [f"{'ROOT':<28} {'ROLE':<16} {'GC POLICY':<20} TTL"]
    for r in DATA_LIFECYCLE:
        ttl = f"{r.ttl_days}d" if r.ttl_days is not None else "-"
        lines.append(f"{r.rel:<28} {r.role.value:<16} {r.gc_policy.value:<20} {ttl}")
    return "\n".join(lines)


__all__ = [
    "Role",
    "GCPolicy",
    "DataRoot",
    "DATA_LIFECYCLE",
    "classify",
    "roots_with_policy",
    "is_safe_to_delete",
    "persistence_report",
]
