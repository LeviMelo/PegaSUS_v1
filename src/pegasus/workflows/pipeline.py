"""End-to-end live pipeline orchestrator (MSD §1.2: D → SHE → EFG → PIRS → O_run).

This is the missing operational spine. Until now PegaSUS had a pile of granular
manual commands (datasus ingest, datasus normalize-*, sidra metadata/plan/extract,
source-artifacts build-manifest, compile) with no driver tying them to a single
``UserIntent``. ``run_live_pipeline`` derives every acquisition parameter from the
intent, fetches the real DATASUS event streams and the SIDRA population
denominator, merges them into ONE materialized-external source manifest, and runs
the real compiler — no fixtures, no development builders, no metadata stand-ins.

The acquisition itself is delegated to the existing real building blocks
(``MicrodatasusClient`` over the R subprocess, ``SidraClient`` over the SIDRA API);
this module is purely the intent→params→merge→compile orchestration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash, sha256_file, sha256_text
from pegasus.core.schemas import UserIntent
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.datasus.manifests import normalize_system
from pegasus.geo.state_panel import GeoScope
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
from pegasus.sidra.metadata import read_normalized_metadata_tables
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.schemas import SIDRAMetadata, SIDRARequest
from pegasus.source_artifacts.contracts import (
    inspect_source_artifact,
    write_source_artifact_manifest,
)
from pegasus.registries.race_bridge import RaceBridgeRegistryError, select_compile_race_bridge_prior
from pegasus.workflows.compile import run_compile


class LivePipelineError(RuntimeError):
    """Raised when the live pipeline cannot acquire or compile real sources."""


# SIDRA 9606 resident-population total-category coordinates (sex/race/age = Total).
SIDRA_POPULATION_TABLE = "9606"
SIDRA_POPULATION_VARIABLE = "93"
SIDRA_POPULATION_TOTAL_CLASSIFICATIONS: dict[str, list[str]] = {
    "86": ["95251"],   # Cor ou raça — Total
    "2": ["6794"],     # Sexo — Total
    "287": ["100362"], # Idade — Total
}


def _population_tensor_requested(intent: UserIntent) -> bool:
    return intent.population_mode in {"independent_population_tensor", "sim_informed_population_tensor"}


def _sidra_population_demographic_strata_classifications(metadata) -> dict[str, list[str]]:
    """Official 9606 demographic strata on the registered non-overlapping basis."""
    from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, map_category

    table = metadata.tables[SIDRA_POPULATION_TABLE]
    selected: dict[str, list[str]] = {}
    for classification_id, axis in (("86", "race"), ("2", "sex"), ("287", "age_group")):
        categories = []
        for category in table.classifications.get(classification_id, []):
            canonical = map_category(axis, "SIDRA", category)
            if canonical in {TOTAL, UNKNOWN}:
                continue
            categories.append(str(category))
        if not categories:
            raise LivePipelineError(
                f"SIDRA 9606 metadata has no registered non-total categories for {axis} "
                f"(classification {classification_id})."
            )
        selected[classification_id] = sorted(categories, key=lambda x: int(x) if x.isdigit() else x)
    return {
        "86": selected["86"],
        "2": selected["2"],
        "287": selected["287"],
    }


@dataclass(frozen=True)
class LivePipelineResult:
    status: str
    run_dir: str | None
    source_manifest: str | None
    datasus_artifacts: list[dict[str, Any]]
    sidra_artifact: dict[str, Any] | None
    compile_result: dict[str, Any] | None
    sidra_artifacts: list[dict[str, Any]] | None = None
    sidra_compendium: dict[str, Any] | None = None
    reason: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_dir": self.run_dir,
            "source_manifest": self.source_manifest,
            "datasus_artifact_count": len(self.datasus_artifacts),
            "sidra_artifact": self.sidra_artifact,
            "sidra_artifact_count": len(self.sidra_artifacts or ([] if self.sidra_artifact is None else [self.sidra_artifact])),
            "sidra_compendium": self.sidra_compendium,
            "reason": self.reason,
        }


def _load_intent(intent_path: str | Path) -> tuple[dict[str, Any], UserIntent]:
    payload = json.loads(Path(intent_path).read_text(encoding="utf-8"))
    return payload, UserIntent.model_validate(payload)


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


def _resolve_systems(intent: UserIntent) -> list[str]:
    excluded = {normalize_system(s) for s in intent.exclude_systems if _is_datasus(s)}
    systems = ["SIM-DO", "SINASC"]
    if "include_cnes_sih" in set(intent.context_policy):
        systems += ["SIH-RD", "CNES-ST"]
    return [s for s in systems if s not in excluded]


def _is_datasus(system: str) -> bool:
    try:
        normalize_system(system)
        return True
    except ValueError:
        return False


def _years_token(intent: UserIntent) -> str:
    return f"{intent.time.start_year}-{intent.time.end_year}"


# Raw→canonical SHE normalizers per system (vectorized Polars decoders).
_NORMALIZERS: dict[str, tuple[str, str]] = {
    "SIM-DO": ("pegasus.datasus.normalize", "normalize_sim_do_events"),
    "SINASC": ("pegasus.datasus.sinasc_normalize", "normalize_sinasc_events"),
    "SIH-RD": ("pegasus.datasus.sih_normalize", "normalize_sih_rd_events"),
    "CNES-ST": ("pegasus.datasus.cnes_normalize", "normalize_cnes_st_events"),
}


def _normalize_datasus(*, system: str, raw_path: Path, out_path: Path, source_manifest_hash: str) -> Path:
    """Decode raw DATASUS columns into the canonical schema the SHE registry
    admits. The R bridge emits raw uppercase columns (DTOBITO, IDADE, CAUSABAS);
    the substrate registry is keyed by canonical names — without this step every
    field is excluded as unknown and the compile produces nothing."""
    import importlib

    module_name, fn_name = _NORMALIZERS[system]
    fn = getattr(importlib.import_module(module_name), fn_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    fn(input_path=raw_path, output_path=out_path, source_manifest_hash=source_manifest_hash)
    return out_path


def _combine_processed_datasus_chunks(*, system: str, requests: list[Any], out_path: Path) -> Path:
    """Combine fetched DATASUS chunks before SHE normalization.

    The live pipeline fetches annual SIM/SINASC and monthly SIH/CNES chunks. The
    compiler consumes source-system artifacts, not individual fetch requests, so
    normalizing one combined processed table per system avoids thousands of
    duplicated batch-boundary operations on multi-year state runs.
    """
    if out_path.exists():
        return out_path
    frames = [pl.read_parquet(request.processed_path) for request in requests]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        pl.DataFrame().write_parquet(out_path)
        return out_path
    pl.concat(frames, how="diagonal_relaxed").write_parquet(out_path)
    return out_path


def _acquire_datasus(
    *,
    systems: list[str],
    uf: str,
    years: str,
    data_root: Path,
    client: MicrodatasusClient | None,
) -> list[dict[str, Any]]:
    client = client or MicrodatasusClient(data_root=str(data_root), manifest_root=data_root / "manifests" / "datasus")
    artifacts: list[dict[str, Any]] = []
    for system in systems:
        batch = client.fetch(system=system, uf=uf, years=years)
        if not batch.ok:
            failed = [r for r in batch.requests if r.status not in {"success", "cached"}]
            if failed:
                reason = "; ".join(
                    f"{r.system} {r.uf} {r.year_start}-{r.month_start or 'NA'}:{r.status}:{r.error_message}"
                    for r in failed[:12]
                )
            else:
                reason = f"datasus fetch not ok for {system}"
            raise LivePipelineError(f"DATASUS acquisition failed for {system} {uf} {years}: {reason}")
        request_hashes = [sha256_file(Path(path)) for path in batch.manifest_paths]
        combined_hash = sha256_text(json.dumps({
            "system": system,
            "uf": uf,
            "years": years,
            "request_manifest_hashes": request_hashes,
        }, ensure_ascii=False, sort_keys=True))
        combined_processed = (
            data_root / "processed" / "datasus_combined" / system
            / f"uf={uf}" / f"years={years}" / combined_hash[:16] / "processed.parquet"
        )
        canonical_path = (
            data_root / "normalized" / "datasus" / system
            / f"uf={uf}" / f"years={years}" / combined_hash[:16] / "canonical.parquet"
        )
        _combine_processed_datasus_chunks(system=system, requests=list(batch.requests), out_path=combined_processed)
        _normalize_datasus(
            system=system,
            raw_path=combined_processed,
            out_path=canonical_path,
            source_manifest_hash=combined_hash,
        )
        artifacts.append(
            inspect_source_artifact(
                path=canonical_path,
                source_system=system,
                artifact_role="processed_events",
                provenance_mode="materialized_external",
                source_manifest_hash=combined_hash,
                manifest_path=batch.manifest_paths[0] if batch.manifest_paths else None,
            )
        )
    return artifacts


def _sidra_population_localities(metadata, uf: str) -> tuple[str, list[str]]:
    """All N6 municipalities of the UF (preferred) or the N3 UF locality.

    Filters the table's official N6 locality list to those whose IBGE code begins
    with the UF's 2-digit code, so the population denominator covers the whole
    state at municipal resolution."""
    cod2 = GeoScope.from_uf(uf).ibge_uf_cod2
    table = metadata.tables[SIDRA_POPULATION_TABLE]
    by_level = table.localities_by_level
    n6 = [str(loc) for loc in by_level.get("N6", []) if str(loc).startswith(str(cod2))]
    if n6:
        return "N6", sorted(n6)
    n3 = [str(loc) for loc in by_level.get("N3", []) if str(loc) == str(cod2)]
    if n3:
        return "N3", n3
    raise LivePipelineError(f"SIDRA 9606 metadata has no N6/N3 localities for UF {uf} (cod2={cod2})")


def _select_population_period(metadata, intent: UserIntent) -> str:
    """Census period for the population denominator: the latest census period at
    or before the intent window end, else the earliest available."""
    periods = sorted(str(p) for p in metadata.tables[SIDRA_POPULATION_TABLE].periods)
    if not periods:
        raise LivePipelineError("SIDRA 9606 metadata declares no periods")
    eligible = [p for p in periods if p.isdigit() and int(p) <= intent.time.end_year]
    return eligible[-1] if eligible else periods[0]


def _ensure_sidra_metadata_tables(
    *,
    table_ids: list[str],
    metadata_dir: Path,
    data_root: Path,
    client: SidraClient | None,
) -> SIDRAMetadata:
    """Read cached SIDRA 9606 metadata, or fetch it live if absent. Makes the
    pipeline a single command rather than requiring a manual metadata step."""
    requested = {str(table_id) for table_id in table_ids}
    existing: SIDRAMetadata | None = None
    if metadata_dir.exists():
        try:
            existing = read_normalized_metadata_tables(metadata_dir)
            if requested <= set(existing.tables):
                return existing
        except Exception:
            existing = None
    from pegasus.sidra.metadata import fetch_official_metadata, write_normalized_metadata_tables

    missing = sorted(requested - (set(existing.tables) if existing is not None else set()), key=lambda x: int(x) if x.isdigit() else x)
    fetched = fetch_official_metadata(
        table_ids=missing,
        client=client or SidraClient(),
        locality_level="N6",
        raw_dir=data_root / "metadata" / "sidra" / "raw",
    )
    tables = {}
    if existing is not None:
        tables.update(existing.tables)
    tables.update(fetched.tables)
    metadata = SIDRAMetadata(tables=tables)
    write_normalized_metadata_tables(metadata, output_dir=metadata_dir)
    return metadata


def _ensure_sidra_metadata(*, metadata_dir: Path, data_root: Path, client: SidraClient | None):
    return _ensure_sidra_metadata_tables(
        table_ids=[SIDRA_POPULATION_TABLE],
        metadata_dir=metadata_dir,
        data_root=data_root,
        client=client,
    )


def _acquire_sidra_population(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> dict[str, Any]:
    metadata = _ensure_sidra_metadata(metadata_dir=metadata_dir, data_root=data_root, client=client)
    if SIDRA_POPULATION_TABLE not in metadata.tables:
        raise LivePipelineError(f"SIDRA metadata is missing required population table {SIDRA_POPULATION_TABLE}")
    locality_level, localities = _sidra_population_localities(metadata, uf)
    period = _select_population_period(metadata, intent)
    request = SIDRARequest(
        table_id=SIDRA_POPULATION_TABLE,
        variables=[SIDRA_POPULATION_VARIABLE],
        periods=[period],
        locality_level=locality_level,
        localities=localities,
        classifications=dict(SIDRA_POPULATION_TOTAL_CLASSIFICATIONS),
    )
    table_metadata = metadata.tables[SIDRA_POPULATION_TABLE]
    metadata_hash = content_hash(table_metadata.model_dump(mode="json"))
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49_900)
    work_dir = data_root / "sidra" / f"population_{uf}_{period}"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = extract_chunk_plan(
        chunks,
        client=client or SidraClient(),
        concurrency=4,
        raw_dir=work_dir / "raw",
        facts_root=work_dir / "facts",
        metadata_hash=metadata_hash,
        unit_by_variable=table_metadata.units_by_variable,
    )
    failures = [r for r in results if r.status != "success"]
    if failures:
        raise LivePipelineError(f"SIDRA population extraction failed: {failures[0].status} ({len(failures)} chunk failures)")
    facts_paths = [Path(r.facts_path) for r in results if r.facts_path]
    if not facts_paths:
        raise LivePipelineError("SIDRA population extraction produced no facts")
    # Compile requires exactly one SIDRA normalized_facts artifact — concatenate
    # the chunk facts into a single facts parquet covering the whole state.
    combined = pl.concat([pl.read_parquet(p) for p in facts_paths], how="vertical_relaxed")
    combined_path = work_dir / "population_facts.parquet"
    combined.write_parquet(combined_path)
    return inspect_source_artifact(
        path=combined_path,
        source_system="SIDRA",
        artifact_role="normalized_facts",
        provenance_mode="materialized_external",
        source_manifest_hash=metadata_hash,
    )


def _acquire_sidra_population_strata(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> dict[str, Any] | None:
    if not _population_tensor_requested(intent):
        return None
    metadata = _ensure_sidra_metadata(metadata_dir=metadata_dir, data_root=data_root, client=client)
    if SIDRA_POPULATION_TABLE not in metadata.tables:
        raise LivePipelineError(f"SIDRA metadata is missing required population table {SIDRA_POPULATION_TABLE}")
    locality_level, localities = _sidra_population_localities(metadata, uf)
    period = _select_population_period(metadata, intent)
    classifications = _sidra_population_demographic_strata_classifications(metadata)
    request = SIDRARequest(
        table_id=SIDRA_POPULATION_TABLE,
        variables=[SIDRA_POPULATION_VARIABLE],
        periods=[period],
        locality_level=locality_level,
        localities=localities,
        classifications=classifications,
    )
    table_metadata = metadata.tables[SIDRA_POPULATION_TABLE]
    metadata_hash = content_hash({
        **table_metadata.model_dump(mode="json"),
        "population_strata_basis": "sex_race_single_year_age_nonoverlapping",
    })
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49_900)
    work_dir = data_root / "sidra" / f"population_strata_demographic_{uf}_{period}"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = extract_chunk_plan(
        chunks,
        client=client or SidraClient(),
        concurrency=4,
        raw_dir=work_dir / "raw",
        facts_root=work_dir / "facts",
        metadata_hash=metadata_hash,
        unit_by_variable=table_metadata.units_by_variable,
    )
    failures = [r for r in results if r.status != "success"]
    if failures:
        raise LivePipelineError(f"SIDRA population_strata extraction failed: {failures[0].status} ({len(failures)} chunk failures)")
    facts_paths = [Path(r.facts_path) for r in results if r.facts_path]
    if not facts_paths:
        raise LivePipelineError("SIDRA population_strata extraction produced no facts")
    combined = pl.concat([pl.read_parquet(p) for p in facts_paths], how="vertical_relaxed")
    combined_path = work_dir / "population_strata_facts.parquet"
    combined.write_parquet(combined_path)
    return inspect_source_artifact(
        path=combined_path,
        source_system="SIDRA",
        artifact_role="population_strata",
        provenance_mode="materialized_external",
        source_manifest_hash=metadata_hash,
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


def _acquire_sidra_compendium_context(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    artifacts: list[dict[str, Any]] = []
    acquired: list[dict[str, Any]] = []
    sidra_client = client or SidraClient()
    for plan in plans:
        request = plan.request()
        table_metadata = metadata.tables[plan.table_id]
        metadata_hash = content_hash({
            **table_metadata.model_dump(mode="json"),
            "compendium_request": plan.as_manifest(),
        })
        chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49_900)
        work_dir = data_root / "sidra" / "context" / f"tier={plan.tier}" / f"uf={uf}" / f"table={plan.table_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        results = extract_chunk_plan(
            chunks,
            client=sidra_client,
            concurrency=4,
            raw_dir=work_dir / "raw",
            facts_root=work_dir / "facts",
            metadata_hash=metadata_hash,
            unit_by_variable=table_metadata.units_by_variable,
        )
        write_extraction_log(results, output_path=work_dir / "extraction_log.json")
        failures = [r for r in results if r.status != "success"]
        if failures:
            raise LivePipelineError(
                f"SIDRA compendium extraction failed for table {plan.table_id}: "
                f"{failures[0].status} ({len(failures)} chunk failures)"
            )
        facts_paths = [Path(r.facts_path) for r in results if r.facts_path]
        if not facts_paths:
            blocked.append(BlockedCompendiumTable(table_id=plan.table_id, reason="extraction produced no facts"))
            continue
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
        artifacts.append(artifact)
        acquired.append({
            **plan.as_manifest(),
            "row_count": artifact.get("row_count"),
            "artifact_path": str(combined_path),
            "metadata_hash": metadata_hash,
        })
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


def _race_bridge_prior_artifact(
    *,
    intent: UserIntent,
    municipality_cod6: str | None,
) -> dict[str, Any] | None:
    if intent.race_tensor_mode != "downstream_bridge":
        return None
    if municipality_cod6 is None:
        raise LivePipelineError("downstream race bridge live pipeline requires a municipality cod6 scope")
    try:
        entry = select_compile_race_bridge_prior(municipality_cod6=municipality_cod6)
        prior = entry.load_prior()
    except RaceBridgeRegistryError as exc:
        raise LivePipelineError(f"Race bridge prior selection failed: {exc}") from exc
    if str(prior.metadata.get("epistemic_status") or "").lower() in {"validation_fixture_only", "fixture", "synthetic_smoke_fixture"}:
        raise LivePipelineError(
            "Configured race bridge prior is validation-only. A calibrated materialized "
            "RACE-BRIDGE emission prior is required for a production live run."
        )
    return inspect_source_artifact(
        path=entry.prior_path,
        source_system="RACE-BRIDGE",
        artifact_role="emission_prior",
        provenance_mode="materialized_external",
        source_manifest_hash=entry.registry_hash,
    )


def plan_live_pipeline(*, intent_path: str | Path) -> dict[str, Any]:
    """Offline resolution of acquisition parameters from intent — no network, no R.

    Lets the intent→params derivation be validated without a live acquisition."""
    _payload, intent = _load_intent(intent_path)
    uf = _resolve_uf(intent)
    from pegasus.geo.uf import resolve_uf_code
    sidra_compendium_plan: dict[str, Any]
    selected = _selected_compendium_tables()
    selection = _compendium_selection_summary(selected)
    try:
        metadata = read_normalized_metadata_tables("data/metadata/sidra/normalized")
        plans, blocked = _plan_sidra_compendium_from_metadata(intent=intent, uf=uf, metadata=metadata)
        sidra_compendium_plan = {
            "enabled": _compendium_enabled(intent),
            **selection,
            "planned_table_count": len(plans),
            "planned_tables": [plan.as_manifest() for plan in plans],
            "blocked": [item.as_manifest() for item in blocked],
            "metadata_source": "cached_normalized_metadata",
        }
    except Exception as exc:
        sidra_compendium_plan = {
            "enabled": _compendium_enabled(intent),
            **selection,
            "planned_table_count": 0,
            "planned_tables": [],
            "blocked": [
                {"table_id": table.table_id, "reason": f"official metadata not available in dry-run cache: {exc}"}
                for table in selected
                if table.table_id != SIDRA_POPULATION_TABLE
            ],
            "metadata_source": "dry_run_no_network",
        }

    return {
        "intent_path": str(intent_path),
        "uf": uf,
        "uf_ibge_cod2": resolve_uf_code(uf).ibge_cod2,
        "years": _years_token(intent),
        "execution_scale": intent.execution_scale,
        "datasus_systems": _resolve_systems(intent),
        "sidra_population": {
            "table_id": SIDRA_POPULATION_TABLE,
            "variable": SIDRA_POPULATION_VARIABLE,
            "classifications": SIDRA_POPULATION_TOTAL_CLASSIFICATIONS,
            "period_selection": f"latest census period <= {intent.time.end_year}",
            "locality_scope": f"all N6 municipalities of UF {uf}",
        },
        "sidra_population_strata": {
            "requested": _population_tensor_requested(intent),
            "basis": "sex_race_single_year_age_nonoverlapping" if _population_tensor_requested(intent) else None,
            "artifact_role": "population_strata" if _population_tensor_requested(intent) else None,
            "reason": (
                "population tensor modes require live disaggregated SIDRA 9606 strata; "
                "sex, SIDRA self-declared race, and non-overlapping single-year age are registry-projected"
            ) if _population_tensor_requested(intent) else "not requested by population_mode",
        },
        "sidra_compendium": sidra_compendium_plan,
        "sidra_projection": {
            "boundary": "SHE context_facts admission",
            "projection_matrix_id": "sidra_context_total_only_v1",
            "total_policy": "total_only_view",
            "hard_aborts": [
                "unmapped_demographic_category",
                "mixed_total_and_non_total_demographic_categories",
                "direct_percentage_projection_without_denominator_recovery",
                "unbounded_high_dimensional_context_request",
            ],
        },
        "stdfm": {
            "gate": "bounded_interpolate only when missing longitudinal support has >=3 temporal points",
            "single_period_policy": "direct_or_cross_sectional_only",
            "latent_context_policy": "dashboard_unsafe_by_default",
        },
        "race_bridge_prior": {
            "required": intent.race_tensor_mode == "downstream_bridge",
            "source_system": "RACE-BRIDGE" if intent.race_tensor_mode == "downstream_bridge" else None,
            "artifact_role": "emission_prior" if intent.race_tensor_mode == "downstream_bridge" else None,
            "fixture_policy": "validation-only priors are rejected by live pipeline and compile",
        },
        "municipality_filter_codes": list(intent.geography.codes),
    }


def run_live_pipeline(
    *,
    intent_path: str | Path,
    data_root: str | Path = "data",
    run_dir: str | Path | None = None,
    sidra_metadata_dir: str | Path = "data/metadata/sidra/normalized",
    datasus_client: MicrodatasusClient | None = None,
    sidra_client: SidraClient | None = None,
    dry_run: bool = False,
) -> LivePipelineResult:
    """Acquire all required live sources, merge one manifest, and compile."""
    data_root = Path(data_root)
    intent_payload, intent = _load_intent(intent_path)
    uf = _resolve_uf(intent)
    systems = _resolve_systems(intent)
    years = _years_token(intent)
    municipality_cod6: str | None = None
    if intent.geography.codes:
        from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6

        municipality_cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)

    if dry_run:
        return LivePipelineResult(
            status="dry_run",
            run_dir=None,
            source_manifest=None,
            datasus_artifacts=[],
            sidra_artifact=None,
            compile_result=None,
            sidra_compendium=None,
            reason=json.dumps(plan_live_pipeline(intent_path=intent_path), sort_keys=True),
        )

    datasus_artifacts = _acquire_datasus(
        systems=systems, uf=uf, years=years, data_root=data_root, client=datasus_client
    )
    sidra_artifact = _acquire_sidra_population(
        intent=intent, uf=uf, data_root=data_root,
        metadata_dir=Path(sidra_metadata_dir), client=sidra_client,
    )
    sidra_strata_artifact = _acquire_sidra_population_strata(
        intent=intent, uf=uf, data_root=data_root,
        metadata_dir=Path(sidra_metadata_dir), client=sidra_client,
    )
    context_artifacts, sidra_compendium = _acquire_sidra_compendium_context(
        intent=intent, uf=uf, data_root=data_root,
        metadata_dir=Path(sidra_metadata_dir), client=sidra_client,
    )
    race_prior_artifact = _race_bridge_prior_artifact(intent=intent, municipality_cod6=municipality_cod6)
    sidra_artifacts = [
        sidra_artifact,
        *([sidra_strata_artifact] if sidra_strata_artifact is not None else []),
        *context_artifacts,
    ]

    intent_hash = sha256_file(Path(intent_path))
    manifest_dir = data_root / "manifests" / "runs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    combined_manifest_path = manifest_dir / f"live_{Path(intent_path).stem}_{intent_hash[:8]}.source_manifest.json"
    write_source_artifact_manifest(
        artifacts=[*datasus_artifacts, *sidra_artifacts, *([race_prior_artifact] if race_prior_artifact is not None else [])],
        output=combined_manifest_path,
    )

    compile_result = run_compile(
        intent_path=intent_path,
        run_dir=run_dir,
        data_root=data_root,
        source_manifest=combined_manifest_path,
        require_materialized_external=True,
    )
    validation = compile_result.get("validation")
    ok = bool(getattr(validation, "ok", False))
    return LivePipelineResult(
        status="success" if ok else "compile_failed",
        run_dir=str(compile_result.get("run_dir")) if compile_result.get("run_dir") else None,
        source_manifest=str(combined_manifest_path),
        datasus_artifacts=datasus_artifacts,
        sidra_artifact=sidra_artifact,
        sidra_artifacts=sidra_artifacts,
        sidra_compendium=sidra_compendium,
        compile_result=compile_result,
        reason=None if ok else "output bundle validation failed",
    )


__all__ = ["run_live_pipeline", "plan_live_pipeline", "LivePipelineResult", "LivePipelineError"]
