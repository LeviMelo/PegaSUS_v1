"""SIDRA population/strata/civil-registry/census-2000 acquisition (MSD §2.8).

Extracted verbatim from ``pegasus.workflows.pipeline`` (the megazord driver) as a
behavior-preserving code motion. This module owns ``LivePipelineError`` and the
SIDRA_* table/variable/classification constants the population denominator, the
demographic strata, the civil-registry vital totals, and the 2000-census strata
all share, plus the SIDRA metadata ensurers (population and metadata acquisition
are mutually dependent, so they live together to keep the import DAG acyclic).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import UserIntent
from pegasus.geo.state_panel import GeoScope
from pegasus.sidra.api import SidraClient
from pegasus.sidra.extract import extract_chunk_plan
from pegasus.sidra.metadata import read_normalized_metadata_tables
from pegasus.sidra.plan import plan_sidra_chunks
from pegasus.sidra.schemas import SIDRAMetadata, SIDRARequest
from pegasus.source_artifacts.contracts import (
    SourceArtifact,
    inspect_source_artifact,
)


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

# SIDRA 6579 ("Estimativas de População", post-censal series): resident-population
# TOTAL only, annual, for the intercensal years 9606 (census years only) doesn't
# cover. See sidra.population_cube.anchor for the verified-live coverage gaps
# (IBGE publishes no 2007 estimate; census years and the immediate post-census
# processing lag are also absent from 6579 -- 9606 covers the former). Neither
# table's gaps are hardcoded here: both tables' own live SIDRA `periods` metadata
# is trusted directly (config/registries/sidra/sidra_stitching.yaml documents the policy).
SIDRA_INTERCENSAL_POPULATION_TABLE = "6579"
SIDRA_INTERCENSAL_POPULATION_VARIABLE = "9324"

# SIDRA 2093 — the 2000 & 2010 census demographic strata (MSD-III §II.4: "table 2093 MUST be added
# to the compendium as an anchor source"). 9606 only carries 2010 + 2022, so 2093 is the ONLY source
# of the 2000 census age×sex×race breakdown — the third census anchor FAL-POP needs for a full-history
# (2000–) cohort-projected tensor. It shares race (clsf 86) and sex (clsf 2) with 9606 but reports age
# as 19 GROUPS (clsf 58) not single-year (clsf 287), plus an urban/rural situation axis (clsf 1) that
# is pinned to Total. Its coarse age brackets are a CTR disaggregation instance (§II.4): the clean
# 19-bracket partition is selected; the finer single-year axis is derived, roll-ups never summed.
SIDRA_CENSUS_2000_STRATA_TABLE = "2093"
SIDRA_CENSUS_2000_STRATA_VARIABLE = "93"
SIDRA_CENSUS_2000_STRATA_RACE_CLSF = "86"
SIDRA_CENSUS_2000_STRATA_SEX_CLSF = "2"
SIDRA_CENSUS_2000_STRATA_AGE_GROUP_CLSF = "58"
SIDRA_CENSUS_2000_STRATA_SITUATION_CLSF = "1"

# SIDRA civil-registry (Registro Civil) vital-statistics tables, total-only, annual,
# 5570 municipalities, 2003-2024. Feed the net-migration residual (MSD §2.8.7):
# NetMig = dPopulation - Births + Deaths. Same IBGE universe as the population
# estimates, so the residual isolates migration rather than cross-system coverage
# gaps (the reason these are preferred over DATASUS SIM/SINASC as the vital source).
# Pre-2003 windows would instead use tables 197/2609-predecessor and 367/368 (not
# wired -- modern runs are covered by 2609/2683).
SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE = "2609"
SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE = "217"
SIDRA_CIVIL_REGISTRY_DEATHS_TABLE = "2683"
SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE = "343"

# SIDRA has NO request-rate limit -- only a per-request CELL cap -- so acquisition is aggressively
# parallel: many concurrent chunk requests, each just under the cell cap. Wide concurrency (was 4,
# which left the pool idle for small UFs).
_SIDRA_CHUNK_CONCURRENCY = 16
# EMPIRICALLY MEASURED cell cap: the IBGE /valores endpoint returns HTTP 500 at 50,000 cells and
# succeeds at 48,000 (probed 2026-07 on table 9606). The true ceiling is <50k; 49_900 is the safe
# margin. (An earlier "95k proven-safe" was fiction inferred from dead code -- 95k-cell chunks 500.)
# plan.py additionally enforces max_localities_per_request=200.
_SIDRA_MAX_CELLS_PER_REQUEST = 49_900


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


def _sidra_population_localities(metadata, uf: str, *, table_id: str = SIDRA_POPULATION_TABLE) -> tuple[str, list[str]]:
    """All N6 municipalities of the UF (preferred) or the N3 UF locality.

    Filters the table's official N6 locality list to those whose IBGE code begins
    with the UF's 2-digit code, so the population denominator covers the whole
    state at municipal resolution. Works for any population-denominator table
    (9606 or the intercensal 6579) — both declare N6 locality support."""
    cod2 = GeoScope.from_uf(uf).ibge_uf_cod2
    table = metadata.tables[table_id]
    by_level = table.localities_by_level
    n6 = [str(loc) for loc in by_level.get("N6", []) if str(loc).startswith(str(cod2))]
    if n6:
        return "N6", sorted(n6)
    n3 = [str(loc) for loc in by_level.get("N3", []) if str(loc) == str(cod2)]
    if n3:
        return "N3", n3
    raise LivePipelineError(f"SIDRA {table_id} metadata has no N6/N3 localities for UF {uf} (cod2={cod2})")


def _all_census_periods(metadata, *, table_id: str = SIDRA_POPULATION_TABLE) -> list[str]:
    """Every census period the census strata table declares — the scope-invariant census anchor set
    (FAL-POP / MSD-III §II.4 "all three censuses MUST be ingested regardless of a query's time
    window" / §VI.1 build-once-slice-many). Independent of any intent window by construction."""
    table = metadata.tables.get(table_id)
    if table is None:
        return []
    return sorted((str(p) for p in table.periods if str(p).isdigit()), key=int)


def _select_population_period(metadata, intent: UserIntent) -> str:
    """Census period for the population denominator: the latest census period at
    or before the intent window end, else the earliest available."""
    periods = sorted(str(p) for p in metadata.tables[SIDRA_POPULATION_TABLE].periods)
    if not periods:
        raise LivePipelineError("SIDRA 9606 metadata declares no periods")
    eligible = [p for p in periods if p.isdigit() and int(p) <= intent.time.end_year]
    return eligible[-1] if eligible else periods[0]


def _select_table_periods_in_window(metadata, table_id: str, intent: UserIntent) -> list[str]:
    """Every period ``table_id`` declares within ``[intent.time.start_year,
    intent.time.end_year]``, sorted ascending. No hardcoded gap-year exclusion --
    whatever years the table's own live SIDRA metadata does not list are simply
    absent from the result (see SIDRA_INTERCENSAL_POPULATION_TABLE docstring)."""
    table = metadata.tables.get(table_id)
    if table is None:
        return []
    return sorted(
        str(p) for p in table.periods
        if str(p).isdigit() and intent.time.start_year <= int(str(p)) <= intent.time.end_year
    )


def _population_period_plan(metadata, intent: UserIntent) -> dict[str, list[str]]:
    """Census (9606) vs intercensal (6579) closure-total periods (MSD §2.8.10 closure source).

    FAL-POP (§VI.1/§II.4): the population tensor is a scope-invariant national + full-history
    foundational asset, so the closure totals span EVERY census (9606: 2010, 2022) and EVERY
    intercensal year (6579) the tables declare — independent of the query's time window; a query
    slices the built tensor. 9606 takes priority on any year both declare. (Was window-limited, which
    truncated the tensor to the query's years — the §VI.1 scope-invariance bug.)"""
    census = _all_census_periods(metadata, table_id=SIDRA_POPULATION_TABLE)
    intercensal = [
        p for p in _all_census_periods(metadata, table_id=SIDRA_INTERCENSAL_POPULATION_TABLE)
        if p not in set(census)
    ]
    if not census and not intercensal:
        raise LivePipelineError(
            f"Neither SIDRA {SIDRA_POPULATION_TABLE} (census) nor {SIDRA_INTERCENSAL_POPULATION_TABLE} "
            f"(intercensal) declares any period."
        )
    return {SIDRA_POPULATION_TABLE: census, SIDRA_INTERCENSAL_POPULATION_TABLE: intercensal}


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
    # The population denominator tables are load-bearing (required); the compendium
    # context tables are best-effort -- a dead/renamed/flaky context table is skipped,
    # not fatal, so a 90-table fetch isn't sunk by one bad table.
    required = {SIDRA_POPULATION_TABLE, SIDRA_INTERCENSAL_POPULATION_TABLE} & set(missing)
    fetched = fetch_official_metadata(
        table_ids=missing,
        client=client or SidraClient(),
        locality_level="N6",
        raw_dir=data_root / "metadata" / "sidra" / "raw",
        required_table_ids=required,
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
        table_ids=[SIDRA_POPULATION_TABLE, SIDRA_INTERCENSAL_POPULATION_TABLE],
        metadata_dir=metadata_dir,
        data_root=data_root,
        client=client,
    )


def _acquire_sidra_population_table(
    *,
    table_id: str,
    variable_id: str,
    classifications: dict[str, list[str]],
    periods: list[str],
    metadata,
    uf: str,
    data_root: Path,
    client: SidraClient | None,
    work_dir_label: str,
) -> Path | None:
    """Acquire one population-denominator table's facts for ``periods``, or None
    if the table has no periods in this window (e.g. an intent entirely inside
    the intercensal gap has no 9606 rows at all, which is expected, not an error)."""
    if not periods:
        return None
    locality_level, localities = _sidra_population_localities(metadata, uf, table_id=table_id)
    request = SIDRARequest(
        table_id=table_id,
        variables=[variable_id],
        periods=periods,
        locality_level=locality_level,
        localities=localities,
        classifications=dict(classifications),
    )
    table_metadata = metadata.tables[table_id]
    metadata_hash = content_hash(table_metadata.model_dump(mode="json"))
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=_SIDRA_MAX_CELLS_PER_REQUEST)
    work_dir = data_root / "sidra" / f"{work_dir_label}_{uf}_{table_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = extract_chunk_plan(
        chunks,
        client=client or SidraClient(),
        concurrency=_SIDRA_CHUNK_CONCURRENCY,
        raw_dir=work_dir / "raw",
        facts_root=work_dir / "facts",
        metadata_hash=metadata_hash,
        unit_by_variable=table_metadata.units_by_variable,
    )
    failures = [r for r in results if r.status != "success"]
    if failures:
        raise LivePipelineError(f"SIDRA {table_id} extraction failed: {failures[0].status} ({len(failures)} chunk failures)")
    facts_paths = [Path(r.facts_path) for r in results if r.facts_path]
    if not facts_paths:
        raise LivePipelineError(f"SIDRA {table_id} extraction produced no facts")
    combined = pl.concat([pl.read_parquet(p) for p in facts_paths], how="vertical_relaxed")
    combined_path = work_dir / "facts.parquet"
    combined.write_parquet(combined_path)
    return combined_path


def _acquire_sidra_population(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> SourceArtifact:
    """Population-total denominator across the intent's full year window: census
    years from 9606 (full sex/race/age matrix, filtered to Total/Total/Total) plus
    intercensal years from 6579 (annual, total-only) -- MSD §2.8.10's E_{s,t}
    closure source. A single combined normalized_facts artifact carries both
    tables' rows; `sidra.population_cube.anchor.load_combined_population_totals_frame`
    stitches them into one per-year panel."""
    metadata = _ensure_sidra_metadata(metadata_dir=metadata_dir, data_root=data_root, client=client)
    plan = _population_period_plan(metadata, intent)
    work_dir = data_root / "sidra" / f"population_{uf}_{intent.time.start_year}_{intent.time.end_year}"
    work_dir.mkdir(parents=True, exist_ok=True)

    census_path = _acquire_sidra_population_table(
        table_id=SIDRA_POPULATION_TABLE, variable_id=SIDRA_POPULATION_VARIABLE,
        classifications=SIDRA_POPULATION_TOTAL_CLASSIFICATIONS, periods=plan[SIDRA_POPULATION_TABLE],
        metadata=metadata, uf=uf, data_root=data_root, client=client, work_dir_label="population_totals",
    )
    intercensal_path = _acquire_sidra_population_table(
        table_id=SIDRA_INTERCENSAL_POPULATION_TABLE, variable_id=SIDRA_INTERCENSAL_POPULATION_VARIABLE,
        classifications={}, periods=plan[SIDRA_INTERCENSAL_POPULATION_TABLE],
        metadata=metadata, uf=uf, data_root=data_root, client=client, work_dir_label="population_totals",
    )
    parts = [p for p in (census_path, intercensal_path) if p is not None]
    combined = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
    combined_path = work_dir / "population_facts.parquet"
    combined.write_parquet(combined_path)
    metadata_hash = content_hash({
        SIDRA_POPULATION_TABLE: metadata.tables[SIDRA_POPULATION_TABLE].model_dump(mode="json") if SIDRA_POPULATION_TABLE in metadata.tables else None,
        SIDRA_INTERCENSAL_POPULATION_TABLE: metadata.tables[SIDRA_INTERCENSAL_POPULATION_TABLE].model_dump(mode="json") if SIDRA_INTERCENSAL_POPULATION_TABLE in metadata.tables else None,
        "population_period_plan": plan,
    })
    return inspect_source_artifact(
        path=combined_path,
        source_system="SIDRA",
        artifact_role="normalized_facts",
        provenance_mode="materialized_external",
        source_manifest_hash=metadata_hash,
    )


def _civil_registry_total_classifications(metadata, table_id: str) -> dict[str, list[str]]:
    """Pin every classification of a civil-registry table to its Total category so
    the request returns one number per municipality-year. SIDRA's Registro Civil
    tables use category id ``0`` for Total across all classifications (compendium-
    verified); ``plan_sidra_chunks`` validates against live metadata and fails loud
    if that ever diverges, so this can't silently produce wrong strata."""
    table = metadata.tables[table_id]
    return {str(clsf_id): ["0"] for clsf_id in table.classifications}


def _acquire_sidra_civil_registry_vital(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> dict[str, SourceArtifact | None]:
    """Acquire SIDRA civil-registry births (2609) and deaths (2683), total-only per
    municipality-year, for the net-migration residual (MSD §2.8.7). Gated on the
    population tensor being requested. Each table is independent: a table absent for
    the window yields ``None`` (the residual falls back to DATASUS SIM/SINASC)."""
    if not _population_tensor_requested(intent):
        return {"births": None, "deaths": None}
    metadata = _ensure_sidra_metadata_tables(
        table_ids=[SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE, SIDRA_CIVIL_REGISTRY_DEATHS_TABLE],
        metadata_dir=metadata_dir, data_root=data_root, client=client,
    )
    out: dict[str, SourceArtifact | None] = {}
    for key, table_id, variable_id, role in (
        ("births", SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE, SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE, "civil_registry_births"),
        ("deaths", SIDRA_CIVIL_REGISTRY_DEATHS_TABLE, SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE, "civil_registry_deaths"),
    ):
        if table_id not in metadata.tables:
            out[key] = None
            continue
        periods = _select_table_periods_in_window(metadata, table_id, intent)
        path = _acquire_sidra_population_table(
            table_id=table_id, variable_id=variable_id,
            classifications=_civil_registry_total_classifications(metadata, table_id),
            periods=periods, metadata=metadata, uf=uf, data_root=data_root, client=client,
            work_dir_label="civil_registry",
        )
        out[key] = None if path is None else inspect_source_artifact(
            path=path,
            source_system="SIDRA",
            artifact_role=role,
            provenance_mode="materialized_external",
            source_manifest_hash=content_hash(metadata.tables[table_id].model_dump(mode="json")),
        )
    return out


def _acquire_sidra_population_strata(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> SourceArtifact | None:
    if not _population_tensor_requested(intent):
        return None
    metadata = _ensure_sidra_metadata(metadata_dir=metadata_dir, data_root=data_root, client=client)
    if SIDRA_POPULATION_TABLE not in metadata.tables:
        raise LivePipelineError(f"SIDRA metadata is missing required population table {SIDRA_POPULATION_TABLE}")
    locality_level, localities = _sidra_population_localities(metadata, uf)
    # FAL-POP / §II.4 / §VI.1: the population tensor is a scope-invariant foundational asset, so ALL
    # census demographic strata (every 9606 period — 2010, 2022; +2093 for 2000 once registered) are
    # ingested regardless of the query's time window. A query SLICES the built tensor; it never
    # re-scopes which censuses anchored it. (Was: only censuses inside the window, which left a
    # 2021-22 query anchored to 2022 alone with no cohort structure between censuses.)
    periods = _all_census_periods(metadata) or [_select_population_period(metadata, intent)]
    classifications = _sidra_population_demographic_strata_classifications(metadata)
    request = SIDRARequest(
        table_id=SIDRA_POPULATION_TABLE,
        variables=[SIDRA_POPULATION_VARIABLE],
        periods=periods,
        locality_level=locality_level,
        localities=localities,
        classifications=classifications,
    )
    table_metadata = metadata.tables[SIDRA_POPULATION_TABLE]
    metadata_hash = content_hash({
        **table_metadata.model_dump(mode="json"),
        "population_strata_basis": "sex_race_single_year_age_nonoverlapping",
        "population_strata_periods": periods,
    })
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=_SIDRA_MAX_CELLS_PER_REQUEST)
    work_dir = data_root / "sidra" / f"population_strata_demographic_{uf}_{'_'.join(periods)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = extract_chunk_plan(
        chunks,
        client=client or SidraClient(),
        concurrency=_SIDRA_CHUNK_CONCURRENCY,
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


def _census_2000_strata_classifications() -> dict[str, list[str]]:
    """Clean-partition classification selection for SIDRA 2093 (§II.4): the 5 canonical races PLUS
    "Sem declaração" (2781, undeclared race) and 2 sexes crossed with the 13 non-overlapping age
    brackets, urban/rural situation pinned to Total. Roll-up brackets (0-14, 15-64, 65+, 70+, 15-19)
    are excluded so nothing is double-counted. The undeclared-race bin (2781) is fetched so it can be
    RECONCILED into the declared races by local composition at build time (§II.5 FAL-POP-RECON) rather
    than silently dropped — without it the 2000 anchor sums short of the enumerated total."""
    from pegasus.sidra.population_cube.census_2000 import CLEAN_AGE_BRACKETS_2093, SIDRA_2093_UNDECLARED_RACE

    return {
        SIDRA_CENSUS_2000_STRATA_RACE_CLSF: ["2776", "2777", "2778", "2779", "2780", SIDRA_2093_UNDECLARED_RACE],
        SIDRA_CENSUS_2000_STRATA_SEX_CLSF: ["4", "5"],
        SIDRA_CENSUS_2000_STRATA_AGE_GROUP_CLSF: list(CLEAN_AGE_BRACKETS_2093.keys()),
        SIDRA_CENSUS_2000_STRATA_SITUATION_CLSF: ["0"],
    }


def _acquire_sidra_census_2000_strata(
    *,
    intent: UserIntent,
    uf: str,
    data_root: Path,
    metadata_dir: Path,
    client: SidraClient | None,
) -> SourceArtifact | None:
    """Acquire the 2000-census age-bracket × sex × race strata from SIDRA 2093 — the third census
    anchor (FAL-POP / §II.4), on 2093's coarse-bracket age axis (disaggregated to single-year at
    build time by ``population_cube.census_2000``). Gated on the population tensor being requested;
    a 2093 outage yields ``None`` (the 2000 anchor is absent, not fatal — the build still has
    2010/2022)."""
    if not _population_tensor_requested(intent):
        return None
    metadata = _ensure_sidra_metadata_tables(
        table_ids=[SIDRA_CENSUS_2000_STRATA_TABLE], metadata_dir=metadata_dir, data_root=data_root, client=client,
    )
    if SIDRA_CENSUS_2000_STRATA_TABLE not in metadata.tables:
        return None
    locality_level, localities = _sidra_population_localities(metadata, uf, table_id=SIDRA_CENSUS_2000_STRATA_TABLE)
    periods = ["2000"]  # the 2000 census; 2093's 2010 is redundant with 9606's single-year 2010.
    classifications = _census_2000_strata_classifications()
    request = SIDRARequest(
        table_id=SIDRA_CENSUS_2000_STRATA_TABLE,
        variables=[SIDRA_CENSUS_2000_STRATA_VARIABLE],
        periods=periods,
        locality_level=locality_level,
        localities=localities,
        classifications=classifications,
    )
    table_metadata = metadata.tables[SIDRA_CENSUS_2000_STRATA_TABLE]
    metadata_hash = content_hash({
        **table_metadata.model_dump(mode="json"),
        "census_2000_strata_basis": "clean_partition_age_brackets_race_sex_situation_total",
        "census_2000_strata_periods": periods,
    })
    chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=_SIDRA_MAX_CELLS_PER_REQUEST)
    work_dir = data_root / "sidra" / f"census_2000_strata_{uf}"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = extract_chunk_plan(
        chunks, client=client or SidraClient(), concurrency=_SIDRA_CHUNK_CONCURRENCY,
        raw_dir=work_dir / "raw", facts_root=work_dir / "facts",
        metadata_hash=metadata_hash, unit_by_variable=table_metadata.units_by_variable,
    )
    failures = [r for r in results if r.status != "success"]
    if failures:
        raise LivePipelineError(f"SIDRA 2093 (2000 census) extraction failed: {failures[0].status} ({len(failures)} chunk failures)")
    facts_paths = [Path(r.facts_path) for r in results if r.facts_path]
    if not facts_paths:
        return None
    combined = pl.concat([pl.read_parquet(p) for p in facts_paths], how="vertical_relaxed")
    combined_path = work_dir / "census_2000_strata_facts.parquet"
    combined.write_parquet(combined_path, compression="zstd")
    return inspect_source_artifact(
        path=combined_path,
        source_system="SIDRA",
        artifact_role="census_2000_strata",
        provenance_mode="materialized_external",
        source_manifest_hash=metadata_hash,
    )


__all__ = [
    "LivePipelineError",
    "SIDRA_POPULATION_TABLE",
    "SIDRA_POPULATION_VARIABLE",
    "SIDRA_POPULATION_TOTAL_CLASSIFICATIONS",
    "SIDRA_INTERCENSAL_POPULATION_TABLE",
    "SIDRA_INTERCENSAL_POPULATION_VARIABLE",
    "SIDRA_CENSUS_2000_STRATA_TABLE",
    "SIDRA_CENSUS_2000_STRATA_VARIABLE",
    "SIDRA_CENSUS_2000_STRATA_RACE_CLSF",
    "SIDRA_CENSUS_2000_STRATA_SEX_CLSF",
    "SIDRA_CENSUS_2000_STRATA_AGE_GROUP_CLSF",
    "SIDRA_CENSUS_2000_STRATA_SITUATION_CLSF",
    "SIDRA_CIVIL_REGISTRY_BIRTHS_TABLE",
    "SIDRA_CIVIL_REGISTRY_BIRTHS_VARIABLE",
    "SIDRA_CIVIL_REGISTRY_DEATHS_TABLE",
    "SIDRA_CIVIL_REGISTRY_DEATHS_VARIABLE",
    "_SIDRA_CHUNK_CONCURRENCY",
    "_SIDRA_MAX_CELLS_PER_REQUEST",
    "_population_tensor_requested",
    "_sidra_population_demographic_strata_classifications",
    "_sidra_population_localities",
    "_all_census_periods",
    "_select_population_period",
    "_select_table_periods_in_window",
    "_population_period_plan",
    "_ensure_sidra_metadata_tables",
    "_ensure_sidra_metadata",
    "_acquire_sidra_population_table",
    "_acquire_sidra_population",
    "_civil_registry_total_classifications",
    "_acquire_sidra_civil_registry_vital",
    "_acquire_sidra_population_strata",
    "_census_2000_strata_classifications",
    "_acquire_sidra_census_2000_strata",
]
