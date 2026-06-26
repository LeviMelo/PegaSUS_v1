from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.registries.demographic_axis import TOTAL, axis_for_classification, map_category
from pegasus.sidra.schemas import SIDRAMetadata, SIDRARequest, SIDRATableMetadata


class SIDRACompendiumError(ValueError):
    """Raised when a compendium table cannot be converted into a legal request."""


@dataclass(frozen=True)
class CompendiumVariable:
    variable_id: str
    unit: str | None
    measure_type: str | None
    default_keep: bool
    description: str | None = None


@dataclass(frozen=True)
class CompendiumClassification:
    classification_id: str
    axis: str | None
    categories: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class CompendiumTable:
    table_id: str
    tier: str
    group: str | None
    variables: tuple[CompendiumVariable, ...]
    classifications: tuple[CompendiumClassification, ...]

    @property
    def default_keep_variables(self) -> tuple[CompendiumVariable, ...]:
        return tuple(v for v in self.variables if v.default_keep and v.variable_id)


@dataclass(frozen=True)
class CompendiumRequestPlan:
    table_id: str
    variables: tuple[str, ...]
    periods: tuple[str, ...]
    locality_level: str
    localities: tuple[str, ...]
    classifications: dict[str, tuple[str, ...]]
    projection_policy: dict[str, Any]

    def request(self) -> SIDRARequest:
        return SIDRARequest(
            table_id=self.table_id,
            variables=list(self.variables),
            periods=list(self.periods),
            locality_level=self.locality_level,
            localities=list(self.localities),
            classifications={k: list(v) for k, v in self.classifications.items()},
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "variables": list(self.variables),
            "periods": list(self.periods),
            "locality_level": self.locality_level,
            "locality_count": len(self.localities),
            "classifications": {k: list(v) for k, v in self.classifications.items()},
            "projection_policy": self.projection_policy,
        }


@dataclass(frozen=True)
class BlockedCompendiumTable:
    table_id: str
    reason: str

    def as_manifest(self) -> dict[str, Any]:
        return {"table_id": self.table_id, "reason": self.reason}


def load_sidra_compendium(path: str | Path = "config/registries/sidra_compendium.json") -> tuple[CompendiumTable, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tables = payload.get("tables") or {}
    order = payload.get("table_order") or sorted(tables)
    out: list[CompendiumTable] = []
    for table_id in order:
        raw = tables.get(str(table_id))
        if not isinstance(raw, dict):
            continue
        variables: list[CompendiumVariable] = []
        for item in raw.get("variables") or []:
            if not isinstance(item, dict):
                continue
            variable_id = item.get("variable_id") or item.get("id")
            if variable_id is None:
                continue
            variables.append(
                CompendiumVariable(
                    variable_id=str(variable_id),
                    unit=None if item.get("unit") is None else str(item.get("unit")),
                    measure_type=None if item.get("type") is None else str(item.get("type")),
                    default_keep=bool(item.get("default_keep")),
                    description=None if item.get("description") is None else str(item.get("description")),
                )
            )
        classifications: list[CompendiumClassification] = []
        for item in raw.get("classifications") or []:
            if not isinstance(item, dict):
                continue
            classification_id = item.get("classification_id") or item.get("id")
            if classification_id is None:
                continue
            cats: list[dict[str, str]] = []
            for cat in item.get("categories") or []:
                if not isinstance(cat, dict):
                    continue
                category_id = cat.get("category_id") or cat.get("id")
                if category_id is None:
                    continue
                cats.append(
                    {
                        "category_id": str(category_id),
                        "label": "" if cat.get("label") is None else str(cat.get("label")),
                    }
                )
            classifications.append(
                CompendiumClassification(
                    classification_id=str(classification_id),
                    axis=None if item.get("axis") is None else str(item.get("axis")),
                    categories=tuple(cats),
                )
            )
        out.append(
            CompendiumTable(
                table_id=str(raw.get("table_id") or table_id),
                tier=str(raw.get("tier") or "T4_LOW_PRIORITY"),
                group=None if raw.get("group") is None else str(raw.get("group")),
                variables=tuple(variables),
                classifications=tuple(classifications),
            )
        )
    return tuple(out)


def select_compendium_tables(
    tables: tuple[CompendiumTable, ...],
    *,
    tiers: tuple[str, ...] = ("T1_CORE",),
) -> tuple[CompendiumTable, ...]:
    allowed = {str(t) for t in tiers}
    return tuple(table for table in tables if table.tier in allowed and table.default_keep_variables)


def _numeric_periods(table: SIDRATableMetadata, *, end_year: int, max_periods: int) -> tuple[str, ...]:
    periods = sorted({str(p) for p in table.periods if str(p).isdigit() and int(str(p)[:4]) <= end_year})
    if not periods:
        raise SIDRACompendiumError("no official numeric periods at or before intent end_year")
    return tuple(periods[-max_periods:])


def _uf_localities(table: SIDRATableMetadata, *, uf_cod2: str) -> tuple[str, str]:
    n6 = tuple(sorted(str(loc) for loc in table.localities_by_level.get("N6", ()) if str(loc).startswith(str(uf_cod2))))
    if n6:
        return "N6", n6
    n3 = tuple(sorted(str(loc) for loc in table.localities_by_level.get("N3", ()) if str(loc) == str(uf_cod2)))
    if n3:
        return "N3", n3
    raise SIDRACompendiumError(f"no N6/N3 localities for UF cod2={uf_cod2}")


def _looks_total(label: str) -> bool:
    return bool(re.search(r"\btotal\b", label.strip().casefold()))


def _registry_total_category(compendium: CompendiumTable, classification_id: str) -> str | None:
    for cls in compendium.classifications:
        if cls.classification_id != str(classification_id):
            continue
        for cat in cls.categories:
            if _looks_total(cat.get("label", "")):
                return str(cat["category_id"])
    return None


def _axis_total_category(classification_id: str, official_categories: list[str]) -> str | None:
    axis = axis_for_classification(str(classification_id))
    if axis is None:
        return None
    for category in official_categories:
        if map_category(axis, "SIDRA", category) == TOTAL:
            return str(category)
    return None


def _legal_classification_request(
    *,
    compendium: CompendiumTable,
    official: SIDRATableMetadata,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    selected: dict[str, tuple[str, ...]] = {}
    policy: dict[str, Any] = {
        "total_category_policy": "total_only_for_context_ingestion",
        "bounded_pushforward": "not_required_total_only",
        "projected_axes": {},
        "blocked_high_dimensional": False,
    }
    for classification_id, official_categories in official.classifications.items():
        total = _registry_total_category(compendium, classification_id)
        if total is None:
            total = _axis_total_category(classification_id, official_categories)
        if total is None:
            raise SIDRACompendiumError(
                f"classification {classification_id} has no registered total category; "
                "refusing unbounded high-dimensional context request"
            )
        if total not in set(map(str, official_categories)):
            raise SIDRACompendiumError(
                f"registered total category {total} is not official for classification {classification_id}"
            )
        selected[str(classification_id)] = (str(total),)
        axis = axis_for_classification(str(classification_id))
        if axis is not None:
            policy["projected_axes"][str(classification_id)] = {
                "axis": axis,
                "category": str(total),
                "canonical": TOTAL,
            }
    return selected, policy


def plan_compendium_request(
    *,
    compendium: CompendiumTable,
    metadata: SIDRAMetadata,
    uf_cod2: str,
    end_year: int,
    max_periods: int = 5,
) -> CompendiumRequestPlan:
    official = metadata.tables.get(compendium.table_id)
    if official is None:
        raise SIDRACompendiumError(f"official metadata missing for table {compendium.table_id}")
    requested_vars = [v.variable_id for v in compendium.default_keep_variables]
    variables = tuple(v for v in requested_vars if v in set(official.variables))
    if not variables:
        raise SIDRACompendiumError("no default-keep variables exist in official metadata")
    periods = _numeric_periods(official, end_year=end_year, max_periods=max_periods)
    locality_level, localities = _uf_localities(official, uf_cod2=uf_cod2)
    classifications, policy = _legal_classification_request(compendium=compendium, official=official)
    return CompendiumRequestPlan(
        table_id=compendium.table_id,
        variables=variables,
        periods=periods,
        locality_level=locality_level,
        localities=localities,
        classifications=classifications,
        projection_policy=policy,
    )


__all__ = [
    "BlockedCompendiumTable",
    "CompendiumRequestPlan",
    "CompendiumTable",
    "SIDRACompendiumError",
    "load_sidra_compendium",
    "plan_compendium_request",
    "select_compendium_tables",
]
