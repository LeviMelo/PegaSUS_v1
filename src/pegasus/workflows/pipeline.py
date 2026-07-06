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

The SIDRA acquisition (population/strata/civil-registry/census-2000, the
socioeconomic compendium, the metadata ensurers, and the national UF fan-out) is
extracted into :mod:`pegasus.workflows.acquire.sidra_population`,
:mod:`pegasus.workflows.acquire.sidra_compendium`,
:mod:`pegasus.workflows.acquire.sidra_metadata`, and
:mod:`pegasus.workflows.acquire.sidra_national`; the symbols are re-imported here
so ``from pegasus.workflows.pipeline import <X>`` keeps working for every caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_file, sha256_text
from pegasus.core.schemas import UserIntent
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.datasus.manifests import normalize_system
from pegasus.sidra.api import SidraClient
from pegasus.sidra.metadata import read_normalized_metadata_tables
from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    inspect_source_artifact,
    write_source_artifact_manifest,
)
from pegasus.registries.race_bridge import RaceBridgeRegistryError, select_compile_race_bridge_prior
from pegasus.workflows.acquire.sidra_compendium import (
    _acquire_one_compendium_table,
    _acquire_sidra_compendium_context,
    _compendium_enabled,
    _compendium_selection_summary,
    _plan_sidra_compendium_from_metadata,
    _selected_compendium_tables,
)
from pegasus.workflows.acquire.sidra_metadata import (
    _ensure_sidra_metadata,
    _ensure_sidra_metadata_tables,
)
from pegasus.workflows.acquire.sidra_national import (
    _acquire_national,
    _acquire_sidra_for_uf,
    _combine_national_artifacts,
    _resolve_uf,
    _resolve_ufs,
)
from pegasus.workflows.acquire.sidra_population import (
    SIDRA_CENSUS_2000_STRATA_AGE_GROUP_CLSF,
    SIDRA_CENSUS_2000_STRATA_RACE_CLSF,
    SIDRA_CENSUS_2000_STRATA_SEX_CLSF,
    SIDRA_CENSUS_2000_STRATA_SITUATION_CLSF,
    SIDRA_CENSUS_2000_STRATA_TABLE,
    SIDRA_CENSUS_2000_STRATA_VARIABLE,
    SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE,
    SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE,
    SIDRA_CIVIL_REGISTRY_DEATHS_TABLE,
    SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE,
    SIDRA_INTERCENSAL_POPULATION_TABLE,
    SIDRA_INTERCENSAL_POPULATION_VARIABLE,
    SIDRA_POPULATION_TABLE,
    SIDRA_POPULATION_TOTAL_CLASSIFICATIONS,
    SIDRA_POPULATION_VARIABLE,
    LivePipelineError,
    _acquire_sidra_census_2000_strata,
    _acquire_sidra_civil_registry_vital,
    _acquire_sidra_population,
    _acquire_sidra_population_strata,
    _all_census_periods,
    _population_tensor_requested,
)
from pegasus.workflows.compile import run_compile


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
    "SINASC": ("pegasus.datasus.normalize", "normalize_sinasc_events"),
    "SIH-RD": ("pegasus.datasus.normalize", "normalize_sih_rd_events"),
    "CNES-ST": ("pegasus.datasus.normalize", "normalize_cnes_st_events"),
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not requests:
        pl.DataFrame().write_parquet(out_path, compression="zstd")
        return out_path
    # Stream the per-chunk scans into one combined artifact via a lazy concat + streaming
    # sink — never materialize all chunks in RAM (§V.1/§V.7(3): no O(national) in-memory
    # copy; a national SIH combine is ~10⁷ rows/year × 27 UFs). ZSTD on write. Mirrors
    # _combine_national_artifacts. Same output file + consumer contract as before.
    lazy = [pl.scan_parquet(request.processed_path) for request in requests]
    pl.concat(lazy, how="diagonal_relaxed").sink_parquet(out_path, compression="zstd")
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
    # Fetch all systems in one global worker pool (parallel across systems AND
    # years) instead of one system at a time.
    batches = client.fetch_systems(systems=systems, uf=uf, years=years)
    # Fail fast (serially) on any unsuccessful fetch before spawning combine/normalize work.
    for system in systems:
        batch = batches[system]
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

    def _combine_normalize_one(system: str) -> dict[str, Any]:
        """Combine chunks → normalize → inspect for one system. Self-contained: writes
        to system-keyed paths, reads only its own already-completed batch, and calls the
        pure ``inspect_source_artifact`` — no shared mutable state, so systems run in
        parallel. The normalizers stream (scan→sink) so per-thread RAM stays bounded."""
        batch = batches[system]
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
        return inspect_source_artifact(
            path=canonical_path,
            source_system=system,
            artifact_role="processed_events",
            provenance_mode="materialized_external",
            source_manifest_hash=combined_hash,
            manifest_path=batch.manifest_paths[0] if batch.manifest_paths else None,
        )

    # Overlap the CPU/I-O-heavy combine+normalize across systems (polars releases the
    # GIL during scan/sink). ``pool.map`` preserves ``systems`` order in the result.
    if len(systems) <= 1:
        return [_combine_normalize_one(system) for system in systems]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(systems), thread_name_prefix="datasus-normalize") as pool:
        return list(pool.map(_combine_normalize_one, systems))


_RACE_BRIDGE_PRIOR_REQUIRED_MODES = frozenset({
    "downstream_bridge", "embedded_fixedC", "embedded_posteriorC", "embedded_sensitivity",
})


def _race_bridge_prior_artifact(
    *,
    intent: UserIntent,
    municipality_cod6: str | None,
) -> SourceArtifact | None:
    if intent.race_tensor_mode not in _RACE_BRIDGE_PRIOR_REQUIRED_MODES:
        return None
    if municipality_cod6 is None:
        raise LivePipelineError("race bridge live pipeline requires a municipality cod6 scope")
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
    # National plans over all 27 UFs; the compendium/locality preview uses the first UF as a
    # representative (per-UF localities are combined at acquisition), so a national intent that
    # declares no single UF still plans without error.
    ufs = _resolve_ufs(intent)
    uf = ufs[0]
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
            "intercensal_table_id": SIDRA_INTERCENSAL_POPULATION_TABLE,
            "intercensal_variable": SIDRA_INTERCENSAL_POPULATION_VARIABLE,
            "period_selection": (
                f"every {SIDRA_POPULATION_TABLE} (census) or {SIDRA_INTERCENSAL_POPULATION_TABLE} "
                f"(intercensal) period in [{intent.time.start_year}, {intent.time.end_year}]; "
                "census wins on overlap (MSD §2.8.10 closure stitching)"
            ),
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
        "sidra_civil_registry_migration": {
            "requested": _population_tensor_requested(intent),
            "births_table_id": SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE if _population_tensor_requested(intent) else None,
            "deaths_table_id": SIDRA_CIVIL_REGISTRY_DEATHS_TABLE if _population_tensor_requested(intent) else None,
            "method": (
                "net-migration residual (MSD §2.8.7): NetMig = dPopulation - Births + Deaths; "
                "SIDRA civil-registry (IBGE-universe-consistent) preferred, DATASUS SIM/SINASC fallback"
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
            "required": intent.race_tensor_mode in _RACE_BRIDGE_PRIOR_REQUIRED_MODES,
            "source_system": "RACE-BRIDGE" if intent.race_tensor_mode in _RACE_BRIDGE_PRIOR_REQUIRED_MODES else None,
            "artifact_role": "emission_prior" if intent.race_tensor_mode in _RACE_BRIDGE_PRIOR_REQUIRED_MODES else None,
            "consumer": (
                "standalone Bridge_R EFG field" if intent.race_tensor_mode == "downstream_bridge"
                else "population tensor birth/death race stratification (MSD §2.8.5/§2.8.6)" if intent.race_tensor_mode in _RACE_BRIDGE_PRIOR_REQUIRED_MODES
                else None
            ),
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
    """Acquire all required live sources, merge one manifest, and compile.

    National scale (SCALE-01) acquires all 27 UFs and combines them into national
    per-(system,role) artifacts before compiling a single national panel."""
    data_root = Path(data_root)
    intent_payload, intent = _load_intent(intent_path)
    ufs = _resolve_ufs(intent)
    national = intent.execution_scale == "national"
    uf = ufs[0]
    systems = _resolve_systems(intent)
    years = _years_token(intent)
    municipality_cod6: str | None = None
    if intent.geography.codes and not national:
        # National declares no single municipality (codes may carry a country token like "BR");
        # a per-municipality cod6 is only meaningful for smoke/state scopes.
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

    if national:
        datasus_artifacts, sidra_artifacts, sidra_compendium = _acquire_national(
            intent=intent, ufs=ufs, systems=systems, years=years, data_root=data_root,
            sidra_metadata_dir=Path(sidra_metadata_dir),
            datasus_client=datasus_client, sidra_client=sidra_client,
        )
        sidra_artifact = next((a for a in sidra_artifacts if a.artifact_role == "normalized_facts"), None)
        race_prior_artifact = None  # national race prior selection is UF-independent; wired later
        all_manifest_artifacts = [*datasus_artifacts, *sidra_artifacts]
        intent_hash = sha256_file(Path(intent_path))
        manifest_dir = data_root / "manifests" / "runs"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        combined_manifest_path = manifest_dir / f"live_{Path(intent_path).stem}_{intent_hash[:8]}.source_manifest.json"
        write_source_artifact_manifest(artifacts=all_manifest_artifacts, output=combined_manifest_path)
        compile_result = run_compile(
            intent_path=intent_path, run_dir=run_dir, data_root=data_root,
            source_manifest=combined_manifest_path, require_materialized_external=True,
        )
        validation = compile_result.get("validation")
        ok = bool(getattr(validation, "ok", False))
        return LivePipelineResult(
            status="success" if ok else "compile_failed",
            run_dir=str(compile_result.get("run_dir")) if compile_result.get("run_dir") else None,
            source_manifest=str(combined_manifest_path),
            datasus_artifacts=datasus_artifacts, sidra_artifact=sidra_artifact,
            sidra_artifacts=sidra_artifacts, sidra_compendium=sidra_compendium,
            compile_result=compile_result, reason=None if ok else "output bundle validation failed",
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
    census_2000_strata_artifact = _acquire_sidra_census_2000_strata(
        intent=intent, uf=uf, data_root=data_root,
        metadata_dir=Path(sidra_metadata_dir), client=sidra_client,
    )
    civil_registry = _acquire_sidra_civil_registry_vital(
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
        *([census_2000_strata_artifact] if census_2000_strata_artifact is not None else []),
        *([civil_registry["births"]] if civil_registry["births"] is not None else []),
        *([civil_registry["deaths"]] if civil_registry["deaths"] is not None else []),
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
