"""National (SCALE-01) UF fan-out and per-(system,role) artifact combination.

Extracted verbatim from ``pegasus.workflows.pipeline``. Owns UF resolution
(``_resolve_uf`` / ``_resolve_ufs``), the per-UF SIDRA acquisition bundle
(``_acquire_sidra_for_uf``), the national driver (``_acquire_national``), and the
streaming per-(system,role) combine (``_combine_national_artifacts``).

``_acquire_national`` also drives DATASUS acquisition, which remains defined in
``pegasus.workflows.pipeline``; it is imported lazily inside the function body to
keep the module import graph acyclic (pipeline imports this module at module load).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_file
from pegasus.core.schemas import UserIntent
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.sidra.api import SidraClient
from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    inspect_source_artifact,
)
from pegasus.workflows.acquire.sidra_compendium import _acquire_sidra_compendium_context
from pegasus.workflows.acquire.sidra_population import (
    SIDRA_CENSUS_2000_STRATA_TABLE,
    SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE,
    SIDRA_CIVIL_REGISTRY_DEATHS_TABLE,
    SIDRA_INTERCENSAL_POPULATION_TABLE,
    SIDRA_POPULATION_TABLE,
    LivePipelineError,
    _acquire_sidra_census_2000_strata,
    _acquire_sidra_civil_registry_vital,
    _acquire_sidra_population,
    _acquire_sidra_population_strata,
    _ensure_sidra_metadata_tables,
)


def _resolve_uf(intent: UserIntent) -> str:
    """The UF whose DATASUS files and SIDRA municipalities must be acquired.

    State scale acquires the declared UF. Smoke (single municipality) still
    acquires the whole UF file — the compiler filters to the municipality."""
    if intent.geography.uf:
        return str(intent.geography.uf[0]).strip().upper()
    if intent.geography.codes:
        from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
        from pegasus.geo.uf import uf_from_datasus_cod6

        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise LivePipelineError(f"cannot resolve DATASUS cod6 for municipality {intent.geography.codes[0]!r}")
        uf = uf_from_datasus_cod6(cod6)
        if uf is None:
            raise LivePipelineError(f"cannot resolve UF for municipality cod6 {cod6!r}")
        return uf
    raise LivePipelineError("intent geography declares neither uf nor municipality codes")


def _resolve_ufs(intent: UserIntent) -> list[str]:
    """The UF(s) to acquire. National scale acquires all 27 federative units; every other
    scale acquires the single declared/derived UF (SCALE-01)."""
    from pegasus.geo.uf import ALL_UF_SIGLAS

    if intent.execution_scale == "national":
        return list(ALL_UF_SIGLAS)
    return [_resolve_uf(intent)]


def _combine_national_artifacts(
    per_uf_artifacts: list[SourceArtifact], *, data_root: Path
) -> list[SourceArtifact]:
    """Combine per-UF source artifacts into one national artifact per (system, role).

    Uses streaming (lazy scan → sink) concat so national SIM/SIH tables (tens of millions
    of rows across 27 states) never have to fit in RAM at once. One national artifact per
    (source_system, artifact_role) feeds the national source manifest.
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for art in per_uf_artifacts:
        groups[(art.source_system, art.artifact_role)].append(Path(art.path))
    ordered_groups = sorted(groups.items())

    def _combine_one(item: tuple[tuple[str, str], list[Path]]) -> SourceArtifact:
        (system, role), paths = item
        out = data_root / "normalized" / "national" / f"{system}__{role}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed").sink_parquet(out, compression="zstd")
        except Exception:
            # sink not available for this frame shape → eager concat fallback
            pl.concat([pl.read_parquet(p) for p in paths], how="diagonal_relaxed").write_parquet(out, compression="zstd")
        return inspect_source_artifact(
            path=out, source_system=system, artifact_role=role,
            provenance_mode="materialized_external", source_manifest_hash=sha256_file(out),
        )

    # Each (system, role) group is an independent streaming sink to a distinct output path;
    # overlap them (polars releases the GIL during scan/sink, per-group RAM stays bounded).
    # ``pool.map`` preserves the sorted group order in the returned artifact list.
    if len(ordered_groups) <= 1:
        return [_combine_one(item) for item in ordered_groups]
    with ThreadPoolExecutor(max_workers=len(ordered_groups), thread_name_prefix="datasus-national-combine") as pool:
        return list(pool.map(_combine_one, ordered_groups))


# How many UFs to fetch SIDRA for CONCURRENTLY. SIDRA has no request-rate limit (only a per-request
# cell cap), so the 27 UFs' chunk requests should overlap rather than serialize -- was fully sequential.
# In-flight requests ~ _SIDRA_UF_PARALLEL x _SIDRA_CHUNK_CONCURRENCY; network-I/O threads (GIL released).
_SIDRA_UF_PARALLEL = 6


def _acquire_sidra_for_uf(
    *, intent: UserIntent, uf: str, data_root: Path, metadata_dir: Path, client: SidraClient,
) -> tuple[list[SourceArtifact], int]:
    """Every SIDRA artifact for one UF (population/strata/2093/civil-registry/compendium), self-
    contained so UFs fetch in parallel. Returns (artifacts, compendium_artifact_count)."""
    arts: list[SourceArtifact] = [
        _acquire_sidra_population(intent=intent, uf=uf, data_root=data_root, metadata_dir=metadata_dir, client=client)
    ]
    strata = _acquire_sidra_population_strata(intent=intent, uf=uf, data_root=data_root, metadata_dir=metadata_dir, client=client)
    if strata is not None:
        arts.append(strata)
    census_2000 = _acquire_sidra_census_2000_strata(intent=intent, uf=uf, data_root=data_root, metadata_dir=metadata_dir, client=client)
    if census_2000 is not None:
        arts.append(census_2000)
    cr = _acquire_sidra_civil_registry_vital(intent=intent, uf=uf, data_root=data_root, metadata_dir=metadata_dir, client=client)
    arts.extend(a for a in (cr.get("births"), cr.get("deaths")) if a is not None)
    ctx, _ctx_summary = _acquire_sidra_compendium_context(intent=intent, uf=uf, data_root=data_root, metadata_dir=metadata_dir, client=client)
    arts.extend(ctx)
    return arts, len(ctx)


def _national_datasus_cached(systems: list[str], data_root: Path) -> list[SourceArtifact] | None:
    """Inspected national DATASUS artifacts iff EVERY requested system already has its
    combined national ``processed_events`` file (§V.7(3): don't re-materialize a cached
    layer). Lets a national re-run skip the per-UF re-acquire+normalize+combine entirely —
    the dominant cost of a warm run (per-UF SIM re-normalize was ~20min / ~19GB). Returns
    ``None`` (do the full per-UF acquire) if any system's national artifact is missing."""
    arts: list[SourceArtifact] = []
    for system in systems:
        out = data_root / "normalized" / "national" / f"{system}__processed_events.parquet"
        if not out.exists():
            return None
        arts.append(
            inspect_source_artifact(
                path=out, source_system=system, artifact_role="processed_events",
                provenance_mode="materialized_external", source_manifest_hash=sha256_file(out),
            )
        )
    return arts


def _acquire_national(
    *, intent: UserIntent, ufs: list[str], systems: list[str], years: str,
    data_root: Path, sidra_metadata_dir: Path,
    datasus_client: MicrodatasusClient | None, sidra_client: SidraClient | None,
) -> tuple[list[SourceArtifact], list[SourceArtifact], dict[str, Any]]:
    """National acquisition (SCALE-01): acquire every UF, then combine into national
    per-(system,role) artifacts. DATASUS runs per UF (R subprocess, internally parallel); SIDRA is
    fetched for all UFs CONCURRENTLY (no rate limit -- only the cell cap) and concatenated.

    Warm-run fast path: if the national DATASUS artifacts already exist they are inspected
    directly (the per-UF re-acquire+normalize+combine is skipped)."""
    from concurrent.futures import ThreadPoolExecutor

    from pegasus.workflows.pipeline import _acquire_datasus

    per_uf_datasus: list[SourceArtifact] = []
    per_uf_sidra: list[SourceArtifact] = []
    compendium_summary: dict[str, Any] = {"status": "national_per_uf_combined", "artifact_count": 0}

    datasus_cached = _national_datasus_cached(systems, data_root) if systems else []
    if datasus_cached is None:
        for uf in ufs:
            per_uf_datasus.extend(
                _acquire_datasus(systems=systems, uf=uf, years=years, data_root=data_root, client=datasus_client)
            )

    # Warm the SIDRA metadata cache ONCE (single-threaded) so the parallel UF fetches only READ it --
    # otherwise 6 workers could race to fetch+write the same table metadata.
    shared_client = sidra_client or SidraClient()
    _ensure_sidra_metadata_tables(
        table_ids=[
            SIDRA_POPULATION_TABLE, SIDRA_INTERCENSAL_POPULATION_TABLE, SIDRA_CENSUS_2000_STRATA_TABLE,
            SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE, SIDRA_CIVIL_REGISTRY_DEATHS_TABLE,
        ],
        metadata_dir=sidra_metadata_dir, data_root=data_root, client=shared_client,
    )
    uf_workers = max(1, min(_SIDRA_UF_PARALLEL, len(ufs))) if ufs else 1
    with ThreadPoolExecutor(max_workers=uf_workers) as pool:
        outcomes = list(pool.map(
            lambda uf: _acquire_sidra_for_uf(
                intent=intent, uf=uf, data_root=data_root, metadata_dir=sidra_metadata_dir, client=shared_client
            ),
            ufs,
        ))
    for arts, ctx_count in outcomes:
        per_uf_sidra.extend(arts)
        compendium_summary["artifact_count"] += ctx_count

    datasus_national = datasus_cached if datasus_cached else _combine_national_artifacts(per_uf_datasus, data_root=data_root)
    if datasus_cached:
        compendium_summary["datasus"] = "national_artifacts_cache_hit"
    sidra_national = _combine_national_artifacts(per_uf_sidra, data_root=data_root)
    return datasus_national, sidra_national, compendium_summary


__all__ = [
    "_resolve_uf",
    "_resolve_ufs",
    "_combine_national_artifacts",
    "_SIDRA_UF_PARALLEL",
    "_acquire_sidra_for_uf",
    "_acquire_national",
]
