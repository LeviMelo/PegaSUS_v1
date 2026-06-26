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

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.core.schemas import UserIntent
from pegasus.datasus.client_microdatasus import MicrodatasusClient
from pegasus.datasus.manifests import normalize_system
from pegasus.geo.state_panel import GeoScope
from pegasus.sidra.api import SidraClient
from pegasus.sidra.extract import extract_chunk_plan
from pegasus.sidra.metadata import read_normalized_metadata_tables
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.schemas import SIDRARequest
from pegasus.source_artifacts.contracts import (
    inspect_source_artifact,
    write_source_artifact_manifest,
)
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


@dataclass(frozen=True)
class LivePipelineResult:
    status: str
    run_dir: str | None
    source_manifest: str | None
    datasus_artifacts: list[dict[str, Any]]
    sidra_artifact: dict[str, Any] | None
    compile_result: dict[str, Any] | None
    reason: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_dir": self.run_dir,
            "source_manifest": self.source_manifest,
            "datasus_artifact_count": len(self.datasus_artifacts),
            "sidra_artifact": self.sidra_artifact,
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
    fn(input_path=raw_path, output_path=out_path, source_manifest_hash=source_manifest_hash)
    return out_path


def _acquire_datasus(
    *,
    systems: list[str],
    uf: str,
    years: str,
    data_root: Path,
    client: MicrodatasusClient | None,
) -> list[dict[str, Any]]:
    client = client or MicrodatasusClient(data_root=str(data_root))
    artifacts: list[dict[str, Any]] = []
    for system in systems:
        batch = client.fetch(system=system, uf=uf, years=years)
        if not batch.ok:
            blocked = [r for r in batch.requests if r.status == "blocked"]
            reason = blocked[0].error_message if blocked else f"datasus fetch not ok for {system}"
            raise LivePipelineError(f"DATASUS acquisition failed for {system} {uf} {years}: {reason}")
        # One request spans the full year range → exactly one processed artifact.
        for request, manifest_path in zip(batch.requests, batch.manifest_paths, strict=True):
            manifest_hash = sha256_file(Path(manifest_path))
            canonical_path = (
                data_root / "normalized" / "datasus" / request.system
                / f"uf={uf}" / f"{manifest_hash[:16]}" / "canonical.parquet"
            )
            _normalize_datasus(
                system=request.system,
                raw_path=Path(request.processed_path),
                out_path=canonical_path,
                source_manifest_hash=manifest_hash,
            )
            artifacts.append(
                inspect_source_artifact(
                    path=canonical_path,
                    source_system=request.system,
                    artifact_role="processed_events",
                    provenance_mode="materialized_external",
                    source_manifest_hash=manifest_hash,
                    manifest_path=manifest_path,
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


def _ensure_sidra_metadata(*, metadata_dir: Path, data_root: Path, client: SidraClient | None):
    """Read cached SIDRA 9606 metadata, or fetch it live if absent. Makes the
    pipeline a single command rather than requiring a manual metadata step."""
    if metadata_dir.exists():
        try:
            metadata = read_normalized_metadata_tables(metadata_dir)
            if SIDRA_POPULATION_TABLE in metadata.tables:
                return metadata
        except Exception:
            pass
    from pegasus.sidra.metadata import fetch_official_metadata, write_normalized_metadata_tables

    metadata = fetch_official_metadata(
        table_ids=[SIDRA_POPULATION_TABLE],
        client=client or SidraClient(),
        locality_level="N6",
        raw_dir=data_root / "metadata" / "sidra" / "raw",
    )
    write_normalized_metadata_tables(metadata, output_dir=metadata_dir)
    return metadata


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


def plan_live_pipeline(*, intent_path: str | Path) -> dict[str, Any]:
    """Offline resolution of acquisition parameters from intent — no network, no R.

    Lets the intent→params derivation be validated without a live acquisition."""
    _payload, intent = _load_intent(intent_path)
    uf = _resolve_uf(intent)
    from pegasus.geo.uf import resolve_uf_code

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

    if dry_run:
        return LivePipelineResult(
            status="dry_run",
            run_dir=None,
            source_manifest=None,
            datasus_artifacts=[],
            sidra_artifact=None,
            compile_result=None,
            reason=json.dumps(plan_live_pipeline(intent_path=intent_path), sort_keys=True),
        )

    datasus_artifacts = _acquire_datasus(
        systems=systems, uf=uf, years=years, data_root=data_root, client=datasus_client
    )
    sidra_artifact = _acquire_sidra_population(
        intent=intent, uf=uf, data_root=data_root,
        metadata_dir=Path(sidra_metadata_dir), client=sidra_client,
    )

    intent_hash = sha256_file(Path(intent_path))
    manifest_dir = data_root / "manifests" / "runs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    combined_manifest_path = manifest_dir / f"live_{Path(intent_path).stem}_{intent_hash[:8]}.source_manifest.json"
    write_source_artifact_manifest(
        artifacts=[*datasus_artifacts, sidra_artifact],
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
        compile_result=compile_result,
        reason=None if ok else "output bundle validation failed",
    )


__all__ = ["run_live_pipeline", "plan_live_pipeline", "LivePipelineResult", "LivePipelineError"]
