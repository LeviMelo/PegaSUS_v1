from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.core.schemas import UserIntent
from pegasus.efg.compile_attach import attach_autonomous_efg_to_run
from pegasus.efg.dag import build_efg
from pegasus.geo.state_panel import GeoScope
from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest
from pegasus.output.bundle_manager import OutputBundleManager
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.race_bridge import RaceBridgePlan, RaceBridgeRegistryError, resolve_race_bridge_plan
from pegasus.efg.race_bridge import load_race_bridge_prior
from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle, load_source_artifacts_from_manifest




def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_intent(intent_path: Path) -> tuple[dict[str, Any], UserIntent]:
    payload = json.loads(intent_path.read_text(encoding="utf-8"))
    try:
        intent = UserIntent.model_validate(payload)
        return intent.model_dump(mode="json"), intent
    except ValidationError as exc:
        raise ValueError(f"Invalid UserIntent file {intent_path}: {exc}") from exc


def _registry_hashes() -> dict[str, str]:
    candidates = [
        Path("config/registries/registry_manifest.yaml"),
        Path("config/registries/sidra/sidra_views.yaml"),
        Path("config/registries/datasus/source_fields.yaml"),
        Path("config/registries/ontology/quality_permissions.yaml"),
        Path("config/registries/demographic/race_axis_registry.yaml"),
        Path("config/registries/demographic/race_bridge_priors.yaml"),
        Path("config/registries/sidra/sidra_compendium.json"),
        Path("config/registries/demographic/demographic_axis_maps.yaml"),
    ]
    return {str(path): sha256_file(path) for path in candidates if path.exists()}


def _write_compile_manifest(*, run_id: str, intent_path: Path, data_root: Path, run_dir: Path, payload: dict[str, Any]) -> Path:
    path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "kind": "compile_manifest",
                "run_id": run_id,
                "intent_path": str(intent_path),
                "run_dir": str(run_dir),
                "intent": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path




def _intent_municipality_filter_cod6(intent: UserIntent) -> str | None:
    if intent.execution_scale == "national":
        # National scope (SCALE-01): materializes a municipality-resolution panel over every
        # UF; the selector level may be 'country' (whole-Brazil) or 'municipality'. No single-
        # municipality filter; validation accepts every real UF via GeoScope.national.
        if intent.geography.codes:
            raise ValueError("National compile expects geography.codes=[] (all municipalities, all UFs).")
        return None

    if intent.geography.level != "municipality":
        raise ValueError("Current compile supports geography.level='municipality' only.")

    if intent.execution_scale == "smoke":
        if len(intent.geography.codes) != 1:
            raise ValueError("Municipality-scale compile requires exactly one municipality code.")
        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise ValueError(
                f"Could not resolve DATASUS cod6 for municipality {intent.geography.codes[0]!r}."
            )
        return cod6

    if intent.execution_scale == "state":
        if intent.geography.codes:
            raise ValueError(
                "State compile expects geography.codes=[] and geography.uf=[<UF>]. "
                "Explicit municipal subsets require a declared panel support contract."
            )
        if len(intent.geography.uf) != 1:
            raise ValueError("State compile requires exactly one UF in geography.uf.")
        return None

    raise ValueError(
        "Compile supports execution_scale='smoke', 'state', or 'national'. "
        f"Received {intent.execution_scale!r}."
    )


def _geo_scope_from_intent(intent: UserIntent, *, municipality_cod6: str | None) -> GeoScope:
    if intent.execution_scale == "national":
        # National always materializes at municipality resolution (all UFs), even when the
        # selector level is declared as 'country'.
        return GeoScope.national(level="municipality")
    if intent.execution_scale == "state":
        if len(intent.geography.uf) != 1:
            raise ValueError("State compile requires exactly one UF in geography.uf.")
        return GeoScope.from_uf(intent.geography.uf[0], level=intent.geography.level)

    if municipality_cod6 is None:
        raise ValueError("Municipality-scale compile requires a resolved DATASUS cod6.")
    return GeoScope(
        level=intent.geography.level,
        uf=None,
        datasus_uf_prefix=str(municipality_cod6)[:2],
        ibge_uf_cod2=str(municipality_cod6)[:2],
        municipality_cod6_allowlist=frozenset({str(municipality_cod6)}),
    )


def _smoke_municipality_cod6(intent: UserIntent) -> str:
    cod6 = _intent_municipality_filter_cod6(intent)
    if cod6 is None:
        raise ValueError("Municipality helper received a state-wide intent.")
    return cod6


def _context_policy_enabled(intent: UserIntent, token: str) -> bool:
    return token in set(intent.context_policy)


def _compile_population_tensor_mode(intent: UserIntent) -> str | None:
    if intent.population_mode == "independent_population_tensor":
        return "independent_denominator"
    if intent.population_mode == "sim_informed_population_tensor":
        return "sim_informed_denominator"
    return None


def _compiler_architecture_metadata() -> dict[str, Any]:
    return {
        "schema_version": "27A.2",
        "graph_authority": "autonomous_efg_core",
        "graph_builder": "pegasus.efg.dag.build_efg",
        "numerical_materialization": "autonomous_compiler_services",
        "numerical_materializer": "pegasus.workflows.compile._run_compile_impl",
        "legacy_bootstrap_builder": None,
        "legacy_bootstrap_status": "retired_deleted",
        "legacy_graph_authority": False,
        "retired_legacy_modules": [],
        "production_runtime_authority": "autonomous_efg_core_plus_compiler_services",
    }


def _seed_user_intent_for_pirs(bundle_manager: Any, run_dir: Path, intent_payload: dict[str, Any]) -> None:
    """Make the real intent (incl. force_selectors) visible to the PIRS stage.

    The schema seed writes a stub UserIntent.json and the canonical serialization
    runs after PIRS, so the PIRS stage workspace would otherwise copy a stub that
    omits force_selectors — silently dropping forced outcome/covariate selection.
    Seeding the bundle's first-class UserIntent payload here ensures the stage
    workspace (write_stage_workspace) carries the forced selectors.
    """
    try:
        bundle_manager.set_json("UserIntent", intent_payload)
    except Exception:
        pass
    p = Path(run_dir) / "UserIntent.json"
    if p.parent.exists():
        p.write_text(
            json.dumps(intent_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )


def _validate_compile_manifest_artifacts(
    artifacts: tuple[SourceArtifactRef, ...],
    *,
    include_cnes_sih: bool,
    run_profile: str = "core_vital",
    require_race_bridge_prior: bool = False,
    excluded_systems: frozenset[str] = frozenset(),
) -> None:
    required: set[tuple[str, str]] = {
        ("SIM-DO", "processed_events"),
        ("SINASC", "processed_events"),
        ("SIDRA", "normalized_facts"),
    }
    if include_cnes_sih:
        required.update({
            ("CNES-ST", "processed_events"),
            ("SIH-RD", "processed_events"),
        })
    if run_profile in {"contextual", "full"}:
        required.add(("SIDRA", "context_facts"))
    if require_race_bridge_prior:
        required.add(("RACE-BRIDGE", "emission_prior"))
    # Honor the intent's declared scope: a system the intent excludes is not a
    # required source artifact (the compile is scope-aware, consistent with
    # exclude_systems and the Run Profile contract).
    if excluded_systems:
        required = {(system, role) for (system, role) in required if system not in excluded_systems}
    available: dict[tuple[str, str], list[SourceArtifactRef]] = {}
    for artifact in artifacts:
        key = (artifact.source_system, artifact.artifact_role)
        available.setdefault(key, []).append(artifact)
        if not Path(artifact.path).is_file():
            raise FileNotFoundError(f"Production source artifact is missing: {artifact.path}")
    for system, role in sorted(required):
        matches = available.get((system, role), [])
        if system == "SIDRA" and role == "normalized_facts":
            if len(matches) != 1:
                raise ValueError(f"Production compile requires exactly one {system}:{role} artifact; found {len(matches)}.")
            continue
        if len(matches) < 1:
            raise ValueError(f"Production compile requires at least one {system}:{role} artifact; found {len(matches)}.")


def _race_bridge_prior_artifact(artifacts: tuple[SourceArtifactRef, ...]) -> SourceArtifactRef | None:
    return next(
        (
            artifact
            for artifact in artifacts
            if artifact.source_system == "RACE-BRIDGE" and artifact.artifact_role == "emission_prior"
        ),
        None,
    )


def _population_tensor_mode_to_solver_mode(population_mode: str) -> str | None:
    if population_mode == "independent_population_tensor":
        return "independent_denominator"
    if population_mode == "sim_informed_population_tensor":
        return "sim_informed_denominator"
    return None


def _max_period_year(facts_path: str | Path, fallback: int) -> int:
    """The latest data year present in the totals facts — the asset's data vintage for its version tag."""
    try:
        value = (
            pl.scan_parquet(str(facts_path))
            .select(pl.col("period").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64, strict=False).max())
            .collect()
            .item()
        )
        return int(value) if value is not None else fallback
    except Exception:
        return fallback


def _build_population_tensor_artifact(
    *,
    artifacts: tuple[SourceArtifactRef, ...],
    run_dir: Path,
    population_mode: str,
    race_bridge_plan: RaceBridgePlan,
    execution_scale: str = "state",
    time_window: Any | None = None,
    asset_store_root: Path | None = None,
) -> tuple[SourceArtifactRef | None, dict[str, Any] | None]:
    solver_mode = _population_tensor_mode_to_solver_mode(population_mode)
    if solver_mode is None:
        return None, None
    total_anchor = next(
        (artifact for artifact in artifacts if artifact.source_system == "SIDRA" and artifact.artifact_role == "normalized_facts"),
        None,
    )
    strata = next(
        (artifact for artifact in artifacts if artifact.source_system == "SIDRA" and artifact.artifact_role == "population_strata"),
        None,
    )
    if total_anchor is None:
        raise ValueError("Population tensor mode requires the SIDRA normalized_facts total-anchor artifact.")
    if strata is None:
        raise ValueError(
            "Population tensor mode requires a real SIDRA population_strata artifact "
            "(disaggregated 9606 facts projected through demographic/demographic_axis_maps.yaml)."
        )
    sim_events = next(
        (artifact for artifact in artifacts if artifact.source_system == "SIM-DO" and artifact.artifact_role == "processed_events"),
        None,
    )
    sinasc_events = next(
        (artifact for artifact in artifacts if artifact.source_system == "SINASC" and artifact.artifact_role == "processed_events"),
        None,
    )
    civil_births = next(
        (artifact for artifact in artifacts if artifact.source_system == "SIDRA" and artifact.artifact_role == "civil_registry_births"),
        None,
    )
    civil_deaths = next(
        (artifact for artifact in artifacts if artifact.source_system == "SIDRA" and artifact.artifact_role == "civil_registry_deaths"),
        None,
    )
    # FAL-POP (§II.4): the 2000-census strata (SIDRA 2093) — the third census anchor. Optional: if
    # absent, the tensor is anchored on 2010/2022 only.
    census_2000 = next(
        (artifact for artifact in artifacts if artifact.source_system == "SIDRA" and artifact.artifact_role == "census_2000_strata"),
        None,
    )
    # Only the "embedded_*" race_tensor_mode feeds the Bridge_R prior into the
    # population tensor's own birth/death race stratification (MSD §2.8.5/§2.8.6);
    # "downstream_bridge" attaches a separate standalone EFG field instead (see
    # registries.race_bridge.resolve_race_bridge_plan), and "decoupled" runs the
    # tensor with race left unstratified for DATASUS-origin priors.
    race_bridge_prior_path = race_bridge_plan.prior_path if race_bridge_plan.status == "embedded" else None
    from pegasus.denominators.population import solve_population_tensor_from_sidra_strata
    from pegasus.assets import (
        NATIONAL_FULL_HISTORY,
        PersistentAssetStore,
        population_tensor_input_identity,
        resolve_or_build_population_tensor,
    )

    # FAL-POP-VER (§VI): the population tensor is a build-once, versioned, scope-invariant foundational
    # asset. A NATIONAL run builds it once at national+full-history scope, stores it immutably, and
    # slices to the query window; a later run whose inputs are unchanged REUSES the stored version
    # rather than rebuilding (the identity below is the build-once key). A reduced-scope (state/dev)
    # build is not stored as foundational — it builds run-locally and slices — so the §VI.1
    # scope-invariance guard is never violated by a query-scope artifact.
    input_hashes = {
        "strata": sha256_file(strata.path),
        "total_anchor": sha256_file(total_anchor.path),
        "census_2000": sha256_file(census_2000.path) if census_2000 is not None else None,
        "sim_events": sha256_file(sim_events.path) if sim_events is not None else None,
        "sinasc_events": sha256_file(sinasc_events.path) if sinasc_events is not None else None,
        "civil_births": sha256_file(civil_births.path) if civil_births is not None else None,
        "civil_deaths": sha256_file(civil_deaths.path) if civil_deaths is not None else None,
        "race_bridge_prior": sha256_file(race_bridge_prior_path) if race_bridge_prior_path else None,
    }
    identity = population_tensor_input_identity(input_hashes=input_hashes, mode=solver_mode)

    def _build_fn(out_path: Path):
        return solve_population_tensor_from_sidra_strata(
            population_strata_path=strata.path,
            total_anchor_path=total_anchor.path,
            output_path=out_path,
            census_2000_strata_path=census_2000.path if census_2000 is not None else None,
            sim_events_path=None if sim_events is None else sim_events.path,
            sinasc_events_path=None if sinasc_events is None else sinasc_events.path,
            civil_registry_births_path=None if civil_births is None else civil_births.path,
            civil_registry_deaths_path=None if civil_deaths is None else civil_deaths.path,
            race_bridge_prior_path=race_bridge_prior_path,
            # Reconstruct the O->D migration flow field + affinity kernel from the net residual
            # (MSD §2.8.7). Bounded: the pair-count guard skips gracefully for scopes too large.
            reconstruct_migration=True,
            mode=solver_mode,
        )

    build_scope = NATIONAL_FULL_HISTORY if execution_scale == "national" else f"{execution_scale}_full_history"
    query_window = (
        (time_window.start_year, time_window.end_year) if time_window is not None else (None, None)
    )
    fallback_year = time_window.end_year if time_window is not None else 2022
    store_root = asset_store_root if asset_store_root is not None else (run_dir.parent.parent / "assets")
    store = PersistentAssetStore(store_root)

    lifecycle = resolve_or_build_population_tensor(
        store=store,
        input_identity=identity,
        build_scope=build_scope,
        build_fn=_build_fn,
        input_manifest={"input_hashes": input_hashes, "mode": solver_mode, "execution_scale": execution_scale},
        max_year=_max_period_year(total_anchor.path, fallback_year),
        query_window=query_window,
        run_dir=run_dir,
        solver_mode=solver_mode,
    )
    sliced_path = lifecycle["sliced_path"]
    manifest = {
        **lifecycle["build_manifest"],
        "asset_name": "population_tensor",
        "asset_version": lifecycle["version"],
        "asset_build_scope": lifecycle["build_scope"],
        "asset_reused": lifecycle["reused"],
        "asset_full_history_path": lifecycle["payload_path"],
        "sliced_to_window": list(query_window),
    }
    artifact = SourceArtifactRef(
        path=str(sliced_path),
        source_system="SIDRA",
        artifact_role="population_tensor",
        provenance_mode="materialized_external",
        source_manifest_hash=strata.source_manifest_hash,
        artifact_hash=sha256_file(sliced_path),
    )
    return artifact, manifest


@dataclass
class _CompilePlan:
    """Resolved compile plan: every local established during plan resolution and
    threaded through the downstream materialization/serialization phases."""

    intent: UserIntent
    intent_payload: dict[str, Any]
    intent_path: Path
    data_root: Path
    source_manifest: str | Path | None
    compile_source_reality: Any
    municipality_cod6: str | None
    geo_scope: GeoScope
    include_cnes_sih: bool
    population_tensor_mode: str | None
    compiler_architecture: dict[str, Any]
    compiler_stage_plan: Any
    race_bridge_plan: RaceBridgePlan
    intent_hash: str
    run_id: str
    run_dir: Path
    telemetry: RunTelemetry
    bundle_manager: OutputBundleManager
    source_hashes: dict[str, str]
    registry_hashes: dict[str, str]
    compile_manifest_path: Path
    autonomous_artifacts: tuple[SourceArtifactRef, ...]


@dataclass
class _StageStatusResult:
    """PURE derivation output: domain-summary metadata + stage-status decisions.

    Holds no telemetry side effects; the ordered `stage_updates` list is applied
    to the telemetry object by `_apply_derived_telemetry`."""

    cnes_sih_metadata: dict[str, Any] | None
    population_tensor_metadata: dict[str, Any] | None
    race_bridge_metadata: dict[str, Any] | None
    sidra_context_metadata: dict[str, Any] | None
    skipped_reasons: dict[str, Any]
    stage_updates: list[tuple[str, str, float]] = field(default_factory=list)


def _resolve_compile_plan(
    *,
    intent_path: Path,
    run_dir: str | Path | None,
    data_root: Path,
    source_manifest: str | Path | None,
    require_materialized_external: bool,
) -> _CompilePlan:
    """Plan resolution: intent, geo scope, race-bridge plan (incl. materialized
    prior validation), stage plan, run identity, telemetry, and manifest-artifact
    validation. Produces the fully-resolved `_CompilePlan` context."""
    from pegasus.source_artifacts.compile_policy import resolve_compile_source_reality

    compile_source_reality = resolve_compile_source_reality(
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    )
    intent_payload, intent = _load_intent(intent_path)
    municipality_cod6 = _intent_municipality_filter_cod6(intent)
    geo_scope = _geo_scope_from_intent(intent, municipality_cod6=municipality_cod6)
    # State-level race bridge: the Bridge_R prior is UF-scoped and its local-pi is computed
    # per-(municipality,year) group at execution, so a whole-UF compile resolves the plan by
    # UF (geo_scope.uf) instead of a single municipality -- enabling self-declared race-
    # specific rates for a state run, not just a single-municipality one.
    race_bridge_uf = geo_scope.uf if municipality_cod6 is None else None
    include_cnes_sih = _context_policy_enabled(intent, "include_cnes_sih")
    population_tensor_mode = _compile_population_tensor_mode(intent)
    compiler_architecture = _compiler_architecture_metadata()
    from pegasus.workflows.stage_plan import build_compile_stage_plan

    compiler_stage_plan = build_compile_stage_plan(
        intent=intent,
        population_tensor_mode=population_tensor_mode,
        include_cnes_sih=include_cnes_sih,
        source_manifest=None if source_manifest is None else str(source_manifest),
        require_materialized_external=require_materialized_external,
    )

    try:
        race_bridge_plan = resolve_race_bridge_plan(
            intent=intent, municipality_cod6=municipality_cod6, uf=race_bridge_uf,
        )
    except RaceBridgeRegistryError as exc:
        raise ValueError(f"Invalid Race Bridge registry plan for {intent_path}: {exc}") from exc
    if race_bridge_plan.status == "blocked":
        raise ValueError(f"Race Bridge mode is blocked for this compile slice: {race_bridge_plan.as_manifest()}")

    intent_hash = sha256_file(intent_path)
    run_id = f"compile_{intent_path.stem}_{utc_stamp()}_{intent_hash[:8]}"
    run_dir = Path(run_dir) if run_dir is not None else data_root / "runs" / run_id
    diagnostic_path = data_root / "diagnostics" / "compile" / f"{run_id}.telemetry.json"
    telemetry = RunTelemetry(run_id=run_id, diagnostic_path=diagnostic_path)
    bundle_manager = OutputBundleManager(run_dir=run_dir)

    source_hashes: dict[str, str] = {"intent": intent_hash}
    registry_hashes = _registry_hashes()
    if race_bridge_plan.registry_path is not None and race_bridge_plan.registry_hash is not None:
        registry_hashes[str(race_bridge_plan.registry_path)] = race_bridge_plan.registry_hash

    compile_manifest_path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
    if source_manifest is None:
        raise ValueError("production compile requires --source-manifest")
    autonomous_artifacts = load_source_artifacts_from_manifest(source_manifest)
    _validate_compile_manifest_artifacts(
        autonomous_artifacts,
        include_cnes_sih=include_cnes_sih,
        run_profile=intent.run_profile,
        require_race_bridge_prior=race_bridge_plan.status in {"planned", "embedded"},
        excluded_systems=frozenset(getattr(intent, "exclude_systems", None) or ()),
    )
    if race_bridge_plan.status in {"planned", "embedded"}:
        prior_artifact = _race_bridge_prior_artifact(autonomous_artifacts)
        if prior_artifact is None:
            raise ValueError("Race bridge compile requires a materialized RACE-BRIDGE:emission_prior artifact.")
        if prior_artifact.provenance_mode != "materialized_external":
            raise ValueError(
                "Race bridge emission prior must be materialized_external; "
                f"received {prior_artifact.provenance_mode!r}."
            )
        prior = load_race_bridge_prior(prior_artifact.path)
        if prior.source_axis != race_bridge_plan.source_axis or prior.target_axis != race_bridge_plan.target_axis:
            raise ValueError(
                "Race bridge emission prior axis mismatch: "
                f"plan={race_bridge_plan.source_axis}->{race_bridge_plan.target_axis}; "
                f"prior={prior.source_axis}->{prior.target_axis}."
            )
        fixture_prior = str(prior.metadata.get("epistemic_status") or "").lower() in {
            "validation_fixture_only", "fixture", "synthetic_smoke_fixture",
        }
        # A fixture/uncalibrated prior may NOT drive a dashboard-safe production run, but it
        # MAY drive an architecture-assessment run: the bridge posterior fields are already
        # emitted as non-dashboard-safe sensitivity observers, so the race-specific rates
        # compute end-to-end while carrying an explicit "uncalibrated" flag and never being
        # presented as calibrated epidemiological truth. Production profiles still refuse it.
        if fixture_prior and str(intent.run_profile) == "full":
            raise ValueError(
                "Race bridge emission prior is marked fixture/validation-only and cannot "
                "drive a production (run_profile='full') compile; register a calibrated prior."
            )
        race_bridge_plan = replace(
            race_bridge_plan,
            bridge_id=prior.bridge_id,
            prior_path=Path(prior_artifact.path),
            prior_hash=prior_artifact.artifact_hash or prior.prior_hash,
            warnings=[
                *list(race_bridge_plan.warnings or []),
                "race_bridge_prior_materialized_external",
                *(["race_bridge_prior_uncalibrated_assessment_only"] if fixture_prior else []),
            ],
        )
        source_hashes["race_bridge_emission_prior"] = prior_artifact.artifact_hash or prior.prior_hash

    return _CompilePlan(
        intent=intent,
        intent_payload=intent_payload,
        intent_path=intent_path,
        data_root=data_root,
        source_manifest=source_manifest,
        compile_source_reality=compile_source_reality,
        municipality_cod6=municipality_cod6,
        geo_scope=geo_scope,
        include_cnes_sih=include_cnes_sih,
        population_tensor_mode=population_tensor_mode,
        compiler_architecture=compiler_architecture,
        compiler_stage_plan=compiler_stage_plan,
        race_bridge_plan=race_bridge_plan,
        intent_hash=intent_hash,
        run_id=run_id,
        run_dir=run_dir,
        telemetry=telemetry,
        bundle_manager=bundle_manager,
        source_hashes=source_hashes,
        registry_hashes=registry_hashes,
        compile_manifest_path=compile_manifest_path,
        autonomous_artifacts=autonomous_artifacts,
    )


def _hash_and_manifest_sources(plan: _CompilePlan) -> None:
    """Source hashing + compile manifest write across the datasus_manifest /
    datasus_acquire / datasus_decode / sidra_normalize telemetry stages. Mutates
    `plan.source_hashes` and `plan.compile_manifest_path` in place."""
    telemetry = plan.telemetry
    source_hashes = plan.source_hashes
    autonomous_artifacts = plan.autonomous_artifacts

    with telemetry.stage("datasus_manifest"):
        plan.compile_manifest_path = _write_compile_manifest(
            run_id=plan.run_id,
            intent_path=plan.intent_path,
            data_root=plan.data_root,
            run_dir=plan.run_dir,
            payload=plan.intent_payload,
        )
        source_hashes["compile_manifest"] = sha256_file(plan.compile_manifest_path)

    with telemetry.stage("datasus_acquire"):
        for idx, artifact in enumerate(autonomous_artifacts):
            digest = artifact.artifact_hash or sha256_file(Path(artifact.path))
            key_base = f"source_artifact_{artifact.source_system}_{artifact.artifact_role}"
            key = key_base if key_base not in source_hashes else f"{key_base}_{idx}_{digest[:8]}"
            source_hashes[key] = digest

    with telemetry.stage("datasus_decode"):
        source_hashes["source_manifest"] = (
            sha256_file(Path(plan.source_manifest))
            if plan.source_manifest is not None
            else content_hash(plan.compile_source_reality.as_manifest())
        )

    telemetry.set_stage("sidra_metadata", "skipped", 0.0)
    telemetry.set_stage("sidra_plan", "skipped", 0.0)
    telemetry.set_stage("sidra_fetch", "skipped", 0.0)
    telemetry.flush()

    with telemetry.stage("sidra_normalize"):
        source_hashes["sidra_facts"] = next(
            artifact.artifact_hash or sha256_file(Path(artifact.path))
            for artifact in autonomous_artifacts
            if artifact.source_system == "SIDRA" and artifact.artifact_role == "normalized_facts"
        )


def _materialize_population_and_substrate(
    plan: _CompilePlan,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    """Population solver + SHE substrate build + autonomous EFG build/attach.

    Mutates `plan.autonomous_artifacts`/`plan.source_hashes` in place and returns
    `(autonomous_result, population_solver_manifest, autonomous_efg_metadata)`."""
    telemetry = plan.telemetry
    intent = plan.intent
    source_hashes = plan.source_hashes

    population_solver_manifest: dict[str, Any] | None = None
    with telemetry.stage("population_solver"):
        population_tensor_artifact, population_solver_manifest = _build_population_tensor_artifact(
            artifacts=plan.autonomous_artifacts,
            run_dir=plan.run_dir,
            population_mode=intent.population_mode,
            race_bridge_plan=plan.race_bridge_plan,
            execution_scale=intent.execution_scale,
            time_window=intent.time,
        )
        if population_tensor_artifact is None:
            telemetry.set_stage("population_solver", "skipped", 0.0)
        else:
            plan.autonomous_artifacts = (*plan.autonomous_artifacts, population_tensor_artifact)
            source_hashes["population_tensor_solver"] = population_tensor_artifact.artifact_hash or sha256_file(Path(population_tensor_artifact.path))

    with telemetry.stage("she_build"):
        autonomous_substrate = build_substrate_bundle(artifacts=plan.autonomous_artifacts)

    with telemetry.stage("efg_build"):
        autonomous_result = build_efg(
            substrate=autonomous_substrate,
            intent=intent,
            intent_constraints={"race_bridge_plan": plan.race_bridge_plan.as_manifest()},
            operator_mode="standard",
        )
        autonomous_attach = attach_autonomous_efg_to_run(
            run_dir=plan.run_dir,
            result=autonomous_result,
            validate=False,
            bundle=plan.bundle_manager,
            intent=intent,
        )
        autonomous_efg_metadata = {
            **autonomous_attach.as_manifest(),
            "substrate_id": autonomous_substrate.substrate_id,
            "source_reality_mode": autonomous_substrate.source_reality_mode,
            "source_systems": sorted({artifact.source_system for artifact in plan.autonomous_artifacts}),
            "registry_hashes": autonomous_result.registry_hashes,
            "legality_summary": autonomous_result.legality_summary,
            "precompression": autonomous_result.precompression.as_manifest(),
        }
        source_hashes["autonomous_efg_manifest"] = autonomous_attach.manifest_hash

    return autonomous_result, population_solver_manifest, autonomous_efg_metadata


def _derive_stage_status(
    *,
    autonomous_result: Any,
    intent: UserIntent,
    compiler_stage_plan: Any,
    population_solver_manifest: dict[str, Any] | None,
) -> _StageStatusResult:
    """PURE (no I/O): extract domain-summary metadata and derive the stage-status
    decisions + skipped-reason map. Emits an ordered `stage_updates` list that
    `_apply_derived_telemetry` replays against the telemetry object."""
    domain_summaries = dict(autonomous_result.domain_summaries or {})
    cnes_sih_metadata: dict[str, Any] | None = domain_summaries.get("cnes_sih")
    population_tensor_metadata: dict[str, Any] | None = domain_summaries.get("population_tensor")
    race_bridge_metadata: dict[str, Any] | None = domain_summaries.get("race_bridge")
    sidra_context_metadata: dict[str, Any] | None = domain_summaries.get("sidra_context")

    stage_updates: list[tuple[str, str, float]] = []
    # MSD cutover: manual domain attachers are forbidden. CNES/SIH, maternal-child,
    # SIDRA denominators, population, and race bridge fields must be produced by
    # SHE substrate + autonomous EFG bridge/operator expansion + physical executor.
    stage_updates.append(("race_bridge", "success" if race_bridge_metadata is not None else "skipped", 0.0))
    stage_updates.append(("geo_support", "success", 0.0))
    stage_updates.append(("q_tensor", "success", 0.0))
    if population_tensor_metadata is None:
        stage_updates.append(("population_solver", "skipped", 0.0))
    stdfm_executed_count = int((sidra_context_metadata or {}).get("stdfm_executed_count") or 0)
    sidra_context_field_count = int((sidra_context_metadata or {}).get("field_count") or 0)
    if stdfm_executed_count > 0:
        stage_updates.append(("stdfm", "success", 0.0))
    elif intent.run_profile in {"contextual", "full"} and sidra_context_field_count > 0:
        stage_updates.append(("stdfm", "skipped", 0.0))
    else:
        stage_updates.append(("stdfm", "skipped", 0.0))
    skipped_reasons = {
        **compiler_stage_plan.skip_reason_map(),
    }
    if intent.run_profile == "core_vital":
        skipped_reasons["stdfm"] = "empty_by_profile: run_profile=core_vital does not require latent-factor fitting"
    elif stdfm_executed_count == 0 and sidra_context_field_count > 0:
        skipped_reasons["stdfm"] = "not_required_by_sidra_regime: context facts admitted without bounded-interpolate reconstruction"
    elif stdfm_executed_count == 0:
        skipped_reasons["stdfm"] = "blocked_no_sidra_context_fields: contextual/full profile requires materialized SIDRA context_facts"
    if population_tensor_metadata is None:
        skipped_reasons["population_solver"] = "official SIDRA anchor selected by intent"

    return _StageStatusResult(
        cnes_sih_metadata=cnes_sih_metadata,
        population_tensor_metadata=population_tensor_metadata,
        race_bridge_metadata=race_bridge_metadata,
        sidra_context_metadata=sidra_context_metadata,
        skipped_reasons=skipped_reasons,
        stage_updates=stage_updates,
    )


def _apply_derived_telemetry(
    plan: _CompilePlan,
    status: _StageStatusResult,
    *,
    population_solver_manifest: dict[str, Any] | None,
) -> None:
    """Replay the PURE stage-status derivation onto the telemetry object,
    preserving the original flush points and resource-summary writes."""
    telemetry = plan.telemetry
    updates = status.stage_updates
    # The first update (race_bridge) is followed by a flush; the remainder are
    # applied together before the skipped_reasons flush — matching the original
    # interleaving of set_stage / flush calls.
    if updates:
        stage, state, value = updates[0]
        telemetry.set_stage(stage, state, value)
        telemetry.flush()
        for stage, state, value in updates[1:]:
            telemetry.set_stage(stage, state, value)
    if status.population_tensor_metadata is not None and population_solver_manifest is not None:
        telemetry.resource_summary["population_solver"] = population_solver_manifest

    # Surface the full intent (incl. force_selectors) into the bundle BEFORE the
    # PIRS stage so the PIRS selection plan honors forced outcome/covariate
    # selectors. The schema-seed writes a stub UserIntent.json that omits them, and
    # the canonical serialization happens after PIRS — without this the PIRS stage
    # workspace copies the stub and the forced outcome is silently dropped.
    # Surface the full intent (incl. force_selectors) into the bundle before output
    # serialization so downstream stages (and the LDO at investigate) see them.
    _seed_user_intent_for_pirs(plan.bundle_manager, plan.run_dir, plan.intent_payload)
    # Inference is an ExecutionStage=investigate concern (MSD-II §II.5). The compile
    # stage materializes V_fields/Q_tensor only; the Lattice Dependency Operator (the
    # LDO, run below after the bundle flush) is the single inference engine, replacing
    # the retired PIRS/HSIC slice-zoo (MII-LDO-06). At compile/validate stage the
    # inference keys stay empty with an empty_by_stage row (anti-silence).
    telemetry.resource_summary["skipped_reasons"] = status.skipped_reasons
    telemetry.flush()


def _serialize_run_bundle(
    plan: _CompilePlan,
    status: _StageStatusResult,
    *,
    autonomous_efg_metadata: dict[str, Any] | None,
) -> None:
    """Run-bundle serialization: RunConfig.json + reproducibility manifests.

    The manifest-extras / final-extras construction (originally two near-identical
    blocks) is factored into a single local `_build_extras` helper."""
    intent = plan.intent
    telemetry = plan.telemetry
    run_dir = plan.run_dir
    run_id = plan.run_id
    cnes_sih_metadata = status.cnes_sih_metadata
    population_tensor_metadata = status.population_tensor_metadata
    race_bridge_metadata = status.race_bridge_metadata

    def _build_extras() -> dict[str, Any]:
        extras = {
            "compile_mode": "compile",
            "run_profile": intent.run_profile,
            "compile_manifest": str(plan.compile_manifest_path),
            "intent_path": str(plan.intent_path),
            "maternal_child_linkage": True,
            "race_bridge_plan": plan.race_bridge_plan.as_manifest(),
            "context_policy": intent.context_policy,
            "compiler_architecture": plan.compiler_architecture,
            "compiler_stage_plan": plan.compiler_stage_plan.as_manifest(),
        }
        if race_bridge_metadata is not None:
            extras["race_bridge"] = race_bridge_metadata
        if cnes_sih_metadata is not None:
            extras["cnes_sih"] = cnes_sih_metadata
        if population_tensor_metadata is not None:
            extras["population_tensor"] = population_tensor_metadata
        if autonomous_efg_metadata is not None:
            extras["autonomous_efg"] = autonomous_efg_metadata
        return extras

    with telemetry.stage("output_serialization"):
        # Ensure the run directory exists for the pre-flush serialization writes. The
        # retired PIRS pipeline used to create it as a side effect; the LDO runs only
        # after the flush, so create it explicitly here (idempotent).
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "UserIntent.json").write_text(
            json.dumps(plan.intent_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        run_config_path = run_dir / "RunConfig.json"
        run_config_payload = {
            "schema_version": "1.0",
            "compile_mode": "compile",
            "run_id": run_id,
            "run_profile": intent.run_profile,
            "intent_path": str(plan.intent_path),
            "data_root": str(plan.data_root),
            "context_policy": intent.context_policy,
            "support_policy": {
                "geography_level": intent.geography.level,
                "ibge_cod7": intent.geography.codes,
                "datasus_cod6": [plan.municipality_cod6],
                "geo_scope": {
                    "level": plan.geo_scope.level,
                    "uf": plan.geo_scope.uf,
                    "datasus_uf_prefix": plan.geo_scope.datasus_uf_prefix,
                    "ibge_uf_cod2": plan.geo_scope.ibge_uf_cod2,
                    "expected_municipality_count": plan.geo_scope.expected_municipality_count,
                },
                "geo_mode": intent.geo_mode,
            },
            "population_mode": intent.population_mode,
            "race_tensor_mode": intent.race_tensor_mode,
            "race_bridge_plan": plan.race_bridge_plan.as_manifest(),
            "compiler_architecture": plan.compiler_architecture,
            "compiler_stage_plan": plan.compiler_stage_plan.as_manifest(),
            "registry_hashes": plan.registry_hashes,
            "source_hashes": plan.source_hashes,
        }
        existing_run_config = {}
        if run_config_path.exists():
            try:
                existing_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_run_config = {}
        if existing_run_config.get("maternal_child_linkage"):
            run_config_payload["maternal_child_linkage"] = existing_run_config["maternal_child_linkage"]
        if race_bridge_metadata is None and existing_run_config.get("race_bridge"):
            race_bridge_metadata = existing_run_config["race_bridge"]
        if race_bridge_metadata is not None:
            run_config_payload["race_bridge"] = race_bridge_metadata
        if cnes_sih_metadata is None and existing_run_config.get("cnes_sih"):
            cnes_sih_metadata = existing_run_config["cnes_sih"]
        if cnes_sih_metadata is not None:
            run_config_payload["cnes_sih"] = cnes_sih_metadata
        if population_tensor_metadata is None and existing_run_config.get("population_tensor"):
            population_tensor_metadata = existing_run_config["population_tensor"]
        if population_tensor_metadata is not None:
            run_config_payload["population_tensor"] = population_tensor_metadata
        if autonomous_efg_metadata is not None:
            run_config_payload["autonomous_efg"] = autonomous_efg_metadata
        run_config_path.write_text(
            json.dumps(run_config_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        write_reproducibility_manifest(
            run_dir=run_dir,
            run_id=run_id,
            intent_hash=plan.intent_hash,
            source_hashes=plan.source_hashes,
            registry_hashes=plan.registry_hashes,
            telemetry=telemetry,
            extras=_build_extras(),
        )

    write_reproducibility_manifest(
        run_dir=run_dir,
        run_id=run_id,
        intent_hash=plan.intent_hash,
        source_hashes=plan.source_hashes,
        registry_hashes=plan.registry_hashes,
        telemetry=telemetry,
        extras=_build_extras(),
    )

    # The RunConfig-merge block above may hydrate metadata from an existing
    # RunConfig.json; write the (possibly-updated) values back onto `status` so the
    # caller's return payload and manifests observe the same values.
    status.race_bridge_metadata = race_bridge_metadata
    status.cnes_sih_metadata = cnes_sih_metadata
    status.population_tensor_metadata = population_tensor_metadata


def _flush_and_investigate(plan: _CompilePlan) -> tuple[Path, dict[str, Any] | None]:
    """Atomic bundle flush + the ExecutionStage=investigate LDO run. Returns the
    (possibly relocated) run_dir and the LDO metadata (or None)."""
    from pegasus.source_artifacts.compile_policy import attach_compile_source_reality

    telemetry = plan.telemetry
    intent = plan.intent
    run_dir = plan.run_dir

    # Phase E boundary: first-class tables are valid only after atomic bundle flush.
    with telemetry.stage("output_bundle_flush"):
        attach_compile_source_reality(run_dir=run_dir, source_reality=plan.compile_source_reality)
        plan.bundle_manager.collect_missing_from_run(run_dir)
        run_dir = plan.bundle_manager.flush_to_disk(run_dir)

    # ExecutionStage=investigate: run the LDO over the freshly-flushed CommonPanel and
    # write the typed LinkRecords to Hypotheses (MSD-II §II.6/§II.8, MII-LDO-06). The
    # LDO needs the panel on disk, so it runs after the flush and before validation.
    ldo_metadata: dict[str, Any] | None = None
    if str(intent.execution_stage) == "investigate":
        from pegasus.workflows.investigate import run_investigate

        inv = run_investigate(run_dir, intent=intent)
        ldo_metadata = {
            "n_link_records": inv.n_link_records,
            "n_selected": inv.n_selected,
            "panel_cells": inv.panel_cells,
            "panel_fields": inv.panel_fields,
            "diagnostics": inv.diagnostics,
        }
        telemetry.resource_summary["ldo"] = ldo_metadata

    return run_dir, ldo_metadata


def _run_compile_impl(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    plan = _resolve_compile_plan(
        intent_path=Path(intent_path),
        run_dir=run_dir,
        data_root=Path(data_root),
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    )

    _hash_and_manifest_sources(plan)

    autonomous_result, population_solver_manifest, autonomous_efg_metadata = (
        _materialize_population_and_substrate(plan)
    )

    status = _derive_stage_status(
        autonomous_result=autonomous_result,
        intent=plan.intent,
        compiler_stage_plan=plan.compiler_stage_plan,
        population_solver_manifest=population_solver_manifest,
    )
    _apply_derived_telemetry(plan, status, population_solver_manifest=population_solver_manifest)

    _serialize_run_bundle(plan, status, autonomous_efg_metadata=autonomous_efg_metadata)

    run_dir, _ldo_metadata = _flush_and_investigate(plan)

    validation = validate_output_bundle(run_dir=str(run_dir))
    return {
        "status": "success" if validation.ok else "failed",
        "run_id": plan.run_id,
        "run_dir": run_dir,
        "intent": plan.intent,
        "validation": validation,
        "source_hashes": plan.source_hashes,
        "registry_hashes": plan.registry_hashes,
        "telemetry": plan.telemetry.model(),
        "race_bridge_plan": plan.race_bridge_plan.as_manifest(),
        "race_bridge": status.race_bridge_metadata,
        "cnes_sih": status.cnes_sih_metadata,
        "population_tensor": status.population_tensor_metadata,
        "autonomous_efg": autonomous_efg_metadata,
        "compiler_architecture": plan.compiler_architecture,
        "compiler_stage_plan": plan.compiler_stage_plan.as_manifest(),
    }

# ---- Slice 13C compile/substrate contract consolidation ----
def run_compile(
    *,
    intent_path: str | Path,
    run_dir: str | Path | None = None,
    data_root: str | Path = "data",
    source_manifest: str | Path | None = None,
    require_materialized_external: bool = False,
) -> dict[str, Any]:
    # Single public compile boundary. All first-class bundle writes must happen
    # inside _run_compile_impl() before OutputBundleManager.flush_to_disk().
    return _run_compile_impl(
        intent_path=intent_path,
        run_dir=run_dir,
        data_root=data_root,
        source_manifest=source_manifest,
        require_materialized_external=require_materialized_external,
    )
