"""Population-tensor foundational-asset lifecycle: build-once, version, slice (§VI, FAL-POP-VER).

The population tensor is a *scope-invariant* foundational asset (§VI.1): built once at national +
full-history scope, stored as an immutable version, and served to every query as a **slice** — never
rebuilt at reduced scope. This module is the lifecycle glue:

- ``population_tensor_input_identity`` — the build-once key (a content hash of the inputs). Unchanged
  inputs => the stored version is reused rather than rebuilt (builds are triggered by data arrival,
  §VI.2, not by queries).
- ``resolve_or_build_population_tensor`` — reuse-or-build + version + slice, in one call.
- ``slice_population_tensor`` — the query view: a lazy ``scan_parquet`` filter to the year window
  (aligned with the §V.7(3) lazy-view direction), never a re-materialization of the whole tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl

from pegasus.assets.store import PersistentAssetStore
from pegasus.assets.version import NATIONAL_FULL_HISTORY, AssetVersion
from pegasus.core.hashing import content_hash

POPULATION_TENSOR = "population_tensor"
# Bump when the tensor build math changes (SV, PROJ, solver): a new code version => a fresh identity
# => a rebuild, so a stored asset never silently reflects stale build logic.
POPULATION_TENSOR_CODE_VERSION = "fal-pop-sv1-proj2-contiguous-years1"


def population_tensor_input_identity(
    *,
    input_hashes: dict[str, str | None],
    mode: str,
    code_version: str = POPULATION_TENSOR_CODE_VERSION,
) -> str:
    """Content hash over the tensor build's inputs + mode + code version — the build-once key."""
    return content_hash({
        "asset": POPULATION_TENSOR,
        "inputs": {k: input_hashes[k] for k in sorted(input_hashes)},
        "mode": mode,
        "code_version": code_version,
    })


def _next_version(store: PersistentAssetStore, name: str, max_year: int) -> str:
    seq = 1 + sum(1 for v in store.history(name) if v.version.startswith(f"v{max_year}."))
    return f"v{max_year}.{seq}"


def slice_population_tensor(
    payload_path: str | Path,
    start_year: int | None,
    end_year: int | None,
    out_path: str | Path,
) -> Path:
    """Slice the full-history tensor parquet to ``[start_year, end_year]`` — a lazy view, not a rebuild.

    ``None`` bounds mean unbounded on that side. Streams via ``scan_parquet``/``sink_parquet`` so a
    national slice never loads the whole tensor into memory."""
    lf = pl.scan_parquet(str(payload_path))
    if start_year is not None:
        lf = lf.filter(pl.col("year") >= int(start_year))
    if end_year is not None:
        lf = lf.filter(pl.col("year") <= int(end_year))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(str(out_path), compression="zstd")
    return out_path


def resolve_or_build_population_tensor(
    *,
    store: PersistentAssetStore,
    input_identity: str,
    build_scope: str,
    build_fn: Callable[[Path], Any],
    input_manifest: dict[str, Any],
    max_year: int,
    query_window: tuple[int | None, int | None],
    run_dir: Path,
    solver_mode: str,
) -> dict[str, Any]:
    """Reuse-or-build the population tensor, then slice it to the query window.

    A NATIONAL + full-history build is stored as an immutable foundational version and reused on any
    later run whose inputs match (``find_by_identity``). A reduced-scope build (state/smoke/dev) is NOT
    stored as a foundational asset — it is built run-locally and sliced — so ``resolve_asset``'s
    scope-invariance guard is never violated by a query-scope artifact masquerading as foundational.

    Returns a dict with ``version``, ``payload_path`` (full-history), ``sliced_path`` (the query view),
    ``reused`` (was the build skipped), ``build_scope``, and ``build_manifest``.
    """
    run_dir = Path(run_dir)
    is_foundational = build_scope == NATIONAL_FULL_HISTORY
    intermediate = run_dir / "Intermediate" / "population_tensor"

    existing = store.find_by_identity(POPULATION_TENSOR, input_identity) if is_foundational else None
    if existing is not None:
        payload_path = Path(existing.payload)
        version = existing.version
        reused = True
        build_manifest = existing.input_manifest.get("build_manifest", {})
    else:
        reused = False
        if is_foundational:
            version = _next_version(store, POPULATION_TENSOR, max_year)
            payload_path = store.version_dir(POPULATION_TENSOR, version) / "population_tensor.parquet"
        else:
            version = "run_local"
            payload_path = intermediate / f"{solver_mode}.full_history.parquet"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        build = build_fn(payload_path)
        build_manifest = build.as_manifest()
        if is_foundational:
            store.put(AssetVersion(
                name=POPULATION_TENSOR,
                version=version,
                build_scope=build_scope,
                input_manifest={
                    **input_manifest,
                    "input_identity": input_identity,
                    "build_manifest": build_manifest,
                },
                certification="unverified",
                payload=str(payload_path),
            ))

    sliced_path = slice_population_tensor(
        payload_path, query_window[0], query_window[1], intermediate / f"{solver_mode}.parquet"
    )
    return {
        "version": version,
        "payload_path": str(payload_path),
        "sliced_path": str(sliced_path),
        "reused": reused,
        "build_scope": build_scope,
        "build_manifest": build_manifest,
    }


__all__ = [
    "POPULATION_TENSOR",
    "POPULATION_TENSOR_CODE_VERSION",
    "population_tensor_input_identity",
    "slice_population_tensor",
    "resolve_or_build_population_tensor",
]
