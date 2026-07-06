"""SIDRA strata parsing: canonical-axis stratum extraction + census-record assembly (MSD §2.8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, map_category


AXES = ("age_group", "sex", "race")
AXIS_CLASSIFICATIONS = {"sex": "2", "race": "86", "age_group": "287"}


def _pairs(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    return [(str(item[0]), str(item[1])) for item in (parsed or []) if len(item) >= 2]


def _category_by_classification(raw: Any) -> dict[str, str]:
    return {classification: category for classification, category in _pairs(raw)}


def _canonical_stratum(row: dict[str, Any]) -> dict[str, str] | None:
    categories = _category_by_classification(row.get("category_tuple"))
    out: dict[str, str] = {}
    for axis, classification_id in AXIS_CLASSIFICATIONS.items():
        raw_code = categories.get(classification_id)
        if raw_code is None:
            out[axis] = TOTAL
            continue
        canonical = map_category(axis, "SIDRA", raw_code)
        if canonical == UNKNOWN:
            return None
        out[axis] = canonical
    return out


def _read_population_strata(path: str | Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    required = {"table_id", "variable_id", "period", "locality_id", "category_tuple", "value_numeric", "value_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"population strata facts missing columns: {sorted(missing)}")
    return frame.filter(
        (pl.col("table_id").cast(pl.Utf8) == "9606")
        & (pl.col("variable_id").cast(pl.Utf8) == "93")
        & (pl.col("value_status").cast(pl.Utf8) == "numeric")
        & pl.col("value_numeric").is_not_null()
    )


def _census_2000_records_from_facts(
    census_2000_path: str | Path, records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Parse SIDRA 2093 (2000-census) facts into single-year age×sex×race records (FAL-POP #4).

    2093 reports age as clsf-58 *brackets*; each (locality, sex, race) bracket profile is CTR-
    disaggregated to single year using that cell's 2010 single-year shape drawn from ``records``
    (the 9606 rows already parsed, which now include 2010 thanks to census-scope invariance).
    Race/sex map through the shared codebook (clsf 86/2) so the 2000 records land on the same axis
    labels as 9606. The **"Sem declaração" (undeclared-race) bin is collected and reconciled into the
    declared races by local composition (§II.5 FAL-POP-RECON), never dropped** — so the 2000 anchor
    sums to the enumerated census total, computationally equal to the 2010/2022 direct-total anchors.
    Returns (records, reconciliation telemetry); ([], {}) on an absent/empty 2093 artifact."""
    from pegasus.denominators.population.census_2000 import (
        CLEAN_AGE_BRACKETS_2093,
        SIDRA_2093_UNDECLARED_RACE,
        assemble_2000_single_year_records,
        reconcile_undeclared_race,
    )

    frame = pl.read_parquet(census_2000_path)
    required = {"table_id", "period", "locality_id", "category_tuple", "value_numeric", "value_status"}
    if required - set(frame.columns):
        return [], {}
    frame = frame.filter(
        (pl.col("table_id").cast(pl.Utf8) == "2093")
        & (pl.col("value_status").cast(pl.Utf8) == "numeric")
        & pl.col("value_numeric").is_not_null()
    )
    if frame.height == 0:
        return [], {}

    profiles: dict[tuple[str, str, str], dict[str, float]] = {}
    undeclared: dict[tuple[str, str], dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        cats = _category_by_classification(row.get("category_tuple"))
        bracket = cats.get("58")
        if bracket not in CLEAN_AGE_BRACKETS_2093:  # skip roll-ups / Total (§II.4: never summed)
            continue
        sex = map_category("sex", "SIDRA", cats.get("2", ""))
        if sex in {TOTAL, UNKNOWN}:
            continue
        loc = str(row["locality_id"])[:6]
        value = float(row["value_numeric"])
        raw_race = str(cats.get("86", ""))
        if raw_race == SIDRA_2093_UNDECLARED_RACE:
            # "Sem declaração": collect by (locality, sex) — reconciled into the declared races below,
            # NOT dropped (the prime directive: missingness is never silence).
            ucell = undeclared.setdefault((loc, sex), {})
            ucell[bracket] = ucell.get(bracket, 0.0) + value
            continue
        race = map_category("race", "SIDRA", raw_race)
        if race in {TOTAL, UNKNOWN}:
            continue
        cell = profiles.setdefault((loc, sex, race), {})
        cell[bracket] = cell.get(bracket, 0.0) + value

    # §II.5 FAL-POP-RECON: reallocate the undeclared-race mass into the declared races by the local
    # declared composition, so the 2000 strata sum to the complete enumerated total (census-exact).
    recon_stats = reconcile_undeclared_race(profiles, undeclared)

    reference_2010: dict[tuple[str, str, str], dict[str, float]] = {}
    for rec in records:
        if rec["period"] == "2010":
            reference_2010.setdefault((rec["municipality_cod6"], rec["sex"], rec["race"]), {})[rec["age_group"]] = rec["value"]

    records_2000 = assemble_2000_single_year_records(bracket_profiles=profiles, reference_2010=reference_2010)
    return records_2000, recon_stats


__all__ = [
    "AXES",
    "AXIS_CLASSIFICATIONS",
    "_pairs",
    "_category_by_classification",
    "_canonical_stratum",
    "_read_population_strata",
    "_census_2000_records_from_facts",
]
