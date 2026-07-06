"""SIDRA socioeconomic compendium context acquisition (MSD context_facts).

Extracted verbatim from ``pegasus.workflows.pipeline``. Acquires the curated
SIDRA compendium's per-table context facts for a UF, concurrently and with
per-table failure isolation. Depends on the SIDRA metadata ensurer and the
shared error/constants from :mod:`sidra_population`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import UserIntent
from pegasus.sidra.api import SidraClient
from pegasus.sidra.compendium import (
    BlockedCompendiumTable,
    CompendiumRequestPlan,
    SIDRACompendiumError,
    load_sidra_compendium,
    plan_compendium_request,
    select_compendium_tables,
)
from pegasus.sidra.extract import extract_chunk_plan, write_extraction_log
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.schemas import SIDRAMetadata
from pegasus.source_artifacts.contracts import inspect_source_artifact
from pegasus.workflows.acquire.sidra_population import (
    SIDRA_POPULATION_TABLE,
    _SIDRA_MAX_CELLS_PER_REQUEST,
    LivePipelineError,
    _ensure_sidra_metadata_tables,
)


def _compendium_enabled(intent: UserIntent) -> bool:
    disabled = {"no_sidra_compendium", "disable_sidra_compendium"}
    if set(intent.context_policy) & disabled:
        return False
    return intent.run_profile in {"contextual", "full"}


def _selected_compendium_tables() -> tuple[Any, ...]:
    return select_compendium_tables(load_sidra_compendium())


def _compendium_selection_summary(selected: tuple[Any, ...]) -> dict[str, Any]:
    tiers: dict[str, int] = {}
    for table in selected:
        tier = str(getattr(table, "tier", "UNKNOWN"))
        tiers[tier] = tiers.get(tier, 0) + 1
    return {
        "scope": "default_keep_catalogue",
        "selected_table_count": len(selected),
        "selected_tiers": dict(sorted(tiers.items())),
    }


def _plan_sidra_compendium_from_metadata(
    *,
    intent: UserIntent,
    uf: str,
    metadata: SIDRAMetadata,
) -> tuple[list[CompendiumRequestPlan], list[BlockedCompendiumTable]]:
    from pegasus.geo.uf import resolve_uf_code

    uf_cod2 = resolve_uf_code(uf).ibge_cod2
    plans: list[CompendiumRequestPlan] = []
    blocked: list[BlockedCompendiumTable] = []
    for table in _selected_compendium_tables():
        if table.table_id == SIDRA_POPULATION_TABLE:
            continue
        try:
            plans.append(
                plan_compendium_request(
                    compendium=table,
                    metadata=metadata,
                    uf_cod2=str(uf_cod2),
                    end_year=int(intent.time.end_year),
                    max_periods=5,
                )
            )
        except SIDRACompendiumError as exc:
            blocked.append(BlockedCompendiumTable(table_id=table.table_id, reason=str(exc)))
    return plans, blocked


def _acquire_one_compendium_table(
    plan: CompendiumRequestPlan,
    *,
    metadata: SIDRAMetadata,
    uf: str,
    data_root: Path,
    sidra_client: SidraClient,
    chunk_concurrency: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, BlockedCompendiumTable | None]:
    """Acquire one compendium table's context facts.

    Returns ``(artifact, acquired_manifest, blocked)`` — exactly one of ``artifact``
    (with manifest) or ``blocked`` is set. A per-table failure is isolated as a
    blocked entry, never an abort of the whole 94-table fetch.
    """
    try:
        request = plan.request()
        table_metadata = metadata.tables[plan.table_id]
        metadata_hash = content_hash({
            **table_metadata.model_dump(mode="json"),
            "compendium_request": plan.as_manifest(),
        })
        chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=_SIDRA_MAX_CELLS_PER_REQUEST)
        work_dir = data_root / "sidra" / "context" / f"tier={plan.tier}" / f"uf={uf}" / f"table={plan.table_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        results = extract_chunk_plan(
            chunks,
            client=sidra_client,
            concurrency=chunk_concurrency,
            raw_dir=work_dir / "raw",
            facts_root=work_dir / "facts",
            metadata_hash=metadata_hash,
            unit_by_variable=table_metadata.units_by_variable,
        )
        write_extraction_log(results, output_path=work_dir / "extraction_log.json")
        failures = [r for r in results if r.status != "success"]
        if failures:
            return None, None, BlockedCompendiumTable(
                table_id=plan.table_id,
                reason=f"extraction failed: {failures[0].status} ({len(failures)} chunk failures)",
            )
        facts_paths = [Path(r.facts_path) for r in results if r.facts_path]
        if not facts_paths:
            return None, None, BlockedCompendiumTable(table_id=plan.table_id, reason="extraction produced no facts")
        combined = pl.concat([pl.read_parquet(p) for p in facts_paths], how="vertical_relaxed")
        combined_path = work_dir / "context_facts.parquet"
        combined.write_parquet(combined_path)
        artifact = inspect_source_artifact(
            path=combined_path,
            source_system="SIDRA",
            artifact_role="context_facts",
            provenance_mode="materialized_external",
            source_manifest_hash=metadata_hash,
        )
        acquired = {
            **plan.as_manifest(),
            "row_count": artifact.as_manifest().get("row_count"),
            "artifact_path": str(combined_path),
            "metadata_hash": metadata_hash,
        }
        return artifact, acquired, None
    except Exception as exc:  # isolate: one bad table must not sink the whole fetch
        return None, None, BlockedCompendiumTable(table_id=plan.table_id, reason=f"{type(exc).__name__}: {exc}")


def _acquire_sidra_compendium_context(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
    table_workers: int = 8,
    chunk_concurrency: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Acquire the curated SIDRA compendium's context facts for ``uf``.

    Ruthless 1-time fetch: tables are acquired concurrently (``table_workers``), each
    over concurrent chunks (``chunk_concurrency``) — SIDRA throttles by cells-per-
    request, not connections, so wide concurrency is safe. Per-table failures are
    isolated (recorded as blocked), so one flaky table never aborts the other 93.
    """
    if not _compendium_enabled(intent):
        return [], {"status": "disabled_by_context_policy", "artifacts": [], "blocked": []}

    selected = _selected_compendium_tables()
    table_ids = sorted(
        {table.table_id for table in selected if table.table_id != SIDRA_POPULATION_TABLE},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if not table_ids:
        return [], {"status": "no_selected_tables", "artifacts": [], "blocked": []}
    metadata = _ensure_sidra_metadata_tables(
        table_ids=[SIDRA_POPULATION_TABLE, *table_ids],
        metadata_dir=metadata_dir,
        data_root=data_root,
        client=client,
    )
    plans, blocked = _plan_sidra_compendium_from_metadata(intent=intent, uf=uf, metadata=metadata)
    sidra_client = client or SidraClient()
    artifacts: list[dict[str, Any]] = []
    acquired: list[dict[str, Any]] = []

    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, min(table_workers, len(plans))) if plans else 1
    if workers <= 1:
        outcomes = [
            _acquire_one_compendium_table(p, metadata=metadata, uf=uf, data_root=data_root, sidra_client=sidra_client, chunk_concurrency=chunk_concurrency)
            for p in plans
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(
                lambda p: _acquire_one_compendium_table(p, metadata=metadata, uf=uf, data_root=data_root, sidra_client=sidra_client, chunk_concurrency=chunk_concurrency),
                plans,
            ))
    for artifact, acquired_manifest, blocked_entry in outcomes:
        if artifact is not None:
            artifacts.append(artifact)
            acquired.append(acquired_manifest)  # type: ignore[arg-type]
        elif blocked_entry is not None:
            blocked.append(blocked_entry)

    if plans and not artifacts:
        raise LivePipelineError(
            "SIDRA compendium acquisition produced no context_facts artifacts; "
            f"blocked={ [b.as_manifest() for b in blocked[:10]] }"
        )
    selection = _compendium_selection_summary(selected)
    return artifacts, {
        "status": "success" if artifacts else "blocked",
        **selection,
        "planned_table_count": len(plans),
        "artifact_count": len(artifacts),
        "artifacts": acquired,
        "blocked": [b.as_manifest() for b in blocked],
    }


__all__ = [
    "_compendium_enabled",
    "_selected_compendium_tables",
    "_compendium_selection_summary",
    "_plan_sidra_compendium_from_metadata",
    "_acquire_one_compendium_table",
    "_acquire_sidra_compendium_context",
]
