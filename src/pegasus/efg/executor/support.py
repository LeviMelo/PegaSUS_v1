"""Physical tensor executor for autonomous EFG.

Operators here execute arrays. Metadata-only operators are not sufficient for
MSD convergence.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

# Allowed cod6 UF prefixes for the current run's geography scope (fix for the
# support-panel pollution: an AL-scoped run must not accumulate national
# municipalities from out-of-scope residence/occurrence codes). None = no filter.
_GEO_SCOPE_PREFIXES: ContextVar["frozenset[str] | None"] = ContextVar("geo_scope_prefixes", default=None)


def _scope_prefixes_from_intent(intent: Any) -> "frozenset[str] | None":
    """Derive the allowed cod6 UF prefixes (2-digit) from a UserIntent or dict."""
    if intent is None:
        return None
    geo = getattr(intent, "geography", None)
    if geo is None and isinstance(intent, dict):
        geo = intent.get("geography")
    if geo is None:
        return None

    def _get(obj: Any, name: str):
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(name)
        return value

    prefixes: set[str] = set()
    for uf in (_get(geo, "uf") or []):
        text = str(uf).strip()
        if text.isdigit() and len(text) >= 2:
            prefixes.add(text[:2])
            continue
        try:
            from pegasus.geo.state_panel import resolve_uf_code

            prefixes.add(str(resolve_uf_code(text).datasus_prefix))
        except Exception:
            continue
    for code in (_get(geo, "codes") or []):
        digits = "".join(ch for ch in str(code) if ch.isdigit())
        if len(digits) >= 2:
            prefixes.add(digits[:2])
    return frozenset(prefixes) or None

from pegasus.core.enums import MaterializationState
from pegasus.core.schemas import FieldNode
from pegasus.efg.dag import EFGResult
from pegasus.measurement.race import (
    bridge_admin_race_group_counts,
    load_race_bridge_prior,
)


VALUE_COLUMN = "value"

METADATA_COLUMNS = {
    "field_id",
    "field_name",
    "operator",
    "source_path",
    "source_column",
}


@dataclass(frozen=True)
class ExecutedField:
    field: FieldNode
    status: str
    path: str | None
    row_count: int
    reason: str | None = None


@dataclass(frozen=True)
class EFGExecutionReport:
    status: str
    executed_count: int
    blocked_count: int
    fields: tuple[ExecutedField, ...]
    output_dir: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "executed_count": self.executed_count,
            "blocked_count": self.blocked_count,
            "output_dir": self.output_dir,
            "fields": [
                {
                    "field_id": item.field.id,
                    "name": item.field.name,
                    "status": item.status,
                    "path": item.path,
                    "row_count": item.row_count,
                    "reason": item.reason,
                }
                for item in self.fields
            ],
        }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _first_existing_path(values: Iterable[Any]) -> Path | None:
    for value in values:
        if value is None:
            continue
        raw = str(value)
        if not raw.lower().endswith(".parquet"):
            continue
        path = Path(raw)
        if path.exists():
            return path
    return None


def _source_path(field: FieldNode) -> Path | None:
    support = _as_dict(field.support)
    candidates = [
        field.path,
        support.get("artifact_path"),
        support.get("source_path"),
        support.get("events_path"),
        support.get("sidra_facts_path"),
        support.get("path"),
        *list(field.source or []),
    ]
    return _first_existing_path(candidates)


def _source_column(field: FieldNode, df: pl.DataFrame) -> str | None:
    support = _as_dict(field.support)
    candidates = [
        support.get("column"),
        support.get("value_column"),
        support.get("source_column"),
        support.get("canonical_field"),
        support.get("raw_field"),
    ]
    candidates.extend(field.source or [])
    for candidate in candidates:
        if candidate is not None and str(candidate) in df.columns:
            return str(candidate)
    return None


def _declared_axis_column(field: FieldNode | None, axis: str) -> str | None:
    if field is None:
        return None
    support = _as_dict(field.support)
    axes = _as_dict(field.axes)
    candidates: list[Any] = []
    if axis == "time":
        candidates.extend([
            support.get("time_column"),
            support.get("year_column"),
            support.get("period_column"),
            support.get("support_time_column"),
        ])
    elif axis == "geography":
        candidates.extend([
            support.get("geography_column"),
            support.get("municipality_column"),
            support.get("support_geography_column"),
        ])
    candidates.extend([
        support.get(f"{axis}_column"),
        support.get("column") if axis in axes else None,
        axes.get(axis),
    ])
    for candidate in candidates:
        if candidate is not None and str(candidate):
            return str(candidate)
    return None


def _frame_columns(frame: "pl.DataFrame | pl.LazyFrame") -> list[str]:
    """Column names for either an eager or lazy frame (schema-only for lazy)."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns


def _with_year_lazy(lf: pl.LazyFrame, field: FieldNode | None = None) -> pl.LazyFrame:
    columns = _frame_columns(lf)
    source = _declared_axis_column(field, "time")
    if source and source in columns:
        return lf.with_columns(pl.col(source).cast(pl.Int64, strict=False).alias("year"))
    if "year" in columns:
        return lf.with_columns(pl.col("year").cast(pl.Int64, strict=False).alias("year"))
    return lf.with_columns(pl.lit(None, dtype=pl.Int64).alias("year"))


def _with_year(df: pl.DataFrame, field: FieldNode | None = None) -> pl.DataFrame:
    return _with_year_lazy(df.lazy(), field).collect()


def _geography_aggregation_of(field: Any) -> str | None:
    """IBGE region level to aggregate event geography to (MSD §3.7), else None."""
    support = _as_dict(getattr(field, "support", {}) or {})
    params = _as_dict(support.get("operator_params") or support.get("params") or {})
    for source in (support, params):
        value = source.get("geography_aggregation")
        if value not in (None, "", "municipality", "none"):
            return str(value)
    return None


def _with_geo_lazy(lf: pl.LazyFrame, field: FieldNode | None = None) -> pl.LazyFrame:
    columns = _frame_columns(lf)
    source = _declared_axis_column(field, "geography")
    if source and source in columns:
        base = lf.with_columns(pl.col(source).cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("municipality_cod6"))
    elif "municipality_cod6" in columns:
        base = lf
    else:
        return lf.with_columns(pl.lit(None, dtype=pl.Utf8).alias("municipality_cod6"))
    base = base.filter(~pl.col("municipality_cod6").cast(pl.Utf8).str.contains(r"^\d{2}0000$").fill_null(False))
    # Geography scope (fix: restrict to the intent's UF prefixes so out-of-scope
    # residence/occurrence municipalities of an e.g. AL-scoped run do not pollute
    # the support lattice with national municipalities). No-op when unset.
    prefixes = _GEO_SCOPE_PREFIXES.get()
    if prefixes:
        base = base.filter(
            pl.col("municipality_cod6").cast(pl.Utf8).str.slice(0, 2).is_in(list(prefixes))
        )
    # Optional spatial aggregation (MSD §3.7): remap the municipality cell to a coarser
    # IBGE region so events accumulate into denser cells. The geography key column name
    # is preserved so all downstream support/RN/HSIC logic is unchanged — only its
    # granularity coarsens. Authoritative crosswalk; unmapped municipalities -> null
    # (dropped by complete-case), never silently mislabelled.
    aggregation = _geography_aggregation_of(field)
    if aggregation:
        from pegasus.geo.region_crosswalk import cod6_to_region_map

        try:
            mapping = cod6_to_region_map(aggregation)
        except Exception:
            mapping = {}
        if mapping:
            map_df = pl.LazyFrame(
                {"municipality_cod6": list(mapping.keys()), "__region_cell__": list(mapping.values())},
                schema={"municipality_cod6": pl.Utf8, "__region_cell__": pl.Utf8},
            )
            base = (
                base.with_columns(pl.col("municipality_cod6").cast(pl.Utf8))
                .join(map_df, on="municipality_cod6", how="left")
                .with_columns(pl.col("__region_cell__").alias("municipality_cod6"))
                .drop("__region_cell__")
            )
    return base


def _with_geo(df: pl.DataFrame, field: FieldNode | None = None) -> pl.DataFrame:
    return _with_geo_lazy(df.lazy(), field).collect()


def _support_frame_lazy(lf: pl.LazyFrame, field: FieldNode | None = None) -> pl.LazyFrame:
    return _with_geo_lazy(_with_year_lazy(lf, field), field)


def _support_frame(df: pl.DataFrame, field: FieldNode | None = None) -> pl.DataFrame:
    return _support_frame_lazy(df.lazy(), field).collect()


def _support_keys_from_columns(columns: Iterable[str]) -> list[str]:
    """Candidate support keys present in a schema (year / municipality_cod6), in order."""
    present = set(columns)
    return [column for column in ("year", "municipality_cod6") if column in present]


def _support_keys(df: pl.DataFrame) -> list[str]:
    keys: list[str] = []
    for column in ("year", "municipality_cod6"):
        if column in df.columns:
            try:
                if bool(df.select(pl.col(column).is_not_null().any()).item()):
                    keys.append(column)
            except Exception:
                pass
    return keys


def _support_keys_lazy(lf: pl.LazyFrame) -> list[str]:
    """Support keys for a lazy frame: drop a candidate key that is entirely null.

    Mirrors the eager `_support_keys` all-null pruning (a key column present in the
    schema but never populated is not a real support axis) with a single cheap
    aggregation pass instead of per-column `.item()` collects."""
    candidates = _support_keys_from_columns(_frame_columns(lf))
    if not candidates:
        return []
    try:
        any_non_null = lf.select(
            [pl.col(column).is_not_null().any().alias(column) for column in candidates]
        ).collect(engine="streaming")
    except Exception:
        return candidates
    return [column for column in candidates if bool(any_non_null[column][0])]


def _write(path: Path, df: pl.DataFrame) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path, df.height


def _materialized(field: FieldNode, path: Path, warnings: list[str] | None = None) -> FieldNode:
    new_warnings = list(dict.fromkeys([
        value for value in [*list(field.warnings or []), *(warnings or [])]
        if value not in {"autonomous_efg_metadata_only", "metadata_only"}
    ]))
    return field.model_copy(update={
        "path": str(path),
        "materialization_state": MaterializationState.materialized,
        "warnings": new_warnings,
    })


def _scalar_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int] | None:
    support = _as_dict(field.support)
    for key in ("value", "n_events", "n_denom", "row_count", "total", "count"):
        if key in support and support[key] is not None:
            try:
                value = float(support[key])
            except Exception:
                continue
            payload: dict[str, list[Any]] = {
                "field_id": [field.id],
                "field_name": [field.name],
                "operator": [field.operator or "scalar_support"],
                VALUE_COLUMN: [value],
            }
            year_value = support.get("year") or support.get("period_year")
            if year_value is None and support.get("period") is not None:
                try:
                    year_value = int(str(support["period"])[:4])
                except Exception:
                    year_value = None
            if year_value is not None:
                payload["year"] = [int(year_value)]
            municipality = support.get("municipality_cod6")
            if municipality is not None:
                payload["municipality_cod6"] = [str(municipality)]
            df = pl.DataFrame(payload)
            return _write(output_dir / f"{field.id}.parquet", df)
    return None


def _icd_stratum_map_df(codes: Iterable[str], icd_column: str, level: str, axis_name: str) -> pl.DataFrame:
    """Build the distinct-code -> group-id crosswalk frame (bounded by the ICD codebook)."""
    from pegasus.datasus.icd_groups import block_for_icd, chapter_for_icd, curated_group_for_icd

    classify = {"chapter": chapter_for_icd, "block": block_for_icd, "curated": curated_group_for_icd}.get(level, chapter_for_icd)
    mapping: dict[str, str] = {}
    for code in codes:
        group = classify(code)
        mapping[str(code)] = group.id if group is not None else "UNCLASSIFIED"
    return pl.DataFrame(
        {icd_column: list(mapping.keys()), axis_name: list(mapping.values())},
        schema={icd_column: pl.Utf8, axis_name: pl.Utf8},
    )


def _add_icd_stratum_lazy(lf: pl.LazyFrame, icd_column: str, level: str, axis_name: str) -> pl.LazyFrame:
    """Lazy variant of `_add_icd_stratum`: map ICD codes to chapter/block ids.

    Distinct codes are pulled via a bounded streaming pass (the codebook has ~14k
    entries — this is not events-scale), then joined back lazily. Identical
    UNCLASSIFIED fallback semantics."""
    if icd_column not in _frame_columns(lf):
        return lf.with_columns(pl.lit("UNCLASSIFIED").alias(axis_name))
    codes = (
        lf.select(pl.col(icd_column).cast(pl.Utf8, strict=False).drop_nulls().unique())
        .collect(engine="streaming")
        .get_column(icd_column)
        .to_list()
    )
    map_df = _icd_stratum_map_df(codes, icd_column, level, axis_name)
    if map_df.height:
        return (
            lf.with_columns(pl.col(icd_column).cast(pl.Utf8, strict=False))
            .join(map_df.lazy(), on=icd_column, how="left")
            .with_columns(pl.col(axis_name).fill_null("UNCLASSIFIED"))
        )
    return lf.with_columns(pl.lit("UNCLASSIFIED").alias(axis_name))


def _add_icd_stratum(df: pl.DataFrame, icd_column: str, level: str, axis_name: str) -> pl.DataFrame:
    """Map each record's ICD code to its chapter/block id (MSD §3.11 σ_C restriction).

    Codes that fall outside the registry-backed groups (or are missing/ill-formed) are
    routed to an explicit ``UNCLASSIFIED`` stratum rather than silently dropped, so the
    cause-specific counts partition the event population exactly.
    """
    if icd_column not in df.columns:
        return df.with_columns(pl.lit("UNCLASSIFIED").alias(axis_name))
    codes = df.get_column(icd_column).cast(pl.Utf8, strict=False).drop_nulls().unique().to_list()
    map_df = _icd_stratum_map_df(codes, icd_column, level, axis_name)
    if map_df.height:
        return (
            df.with_columns(pl.col(icd_column).cast(pl.Utf8, strict=False))
            .join(map_df, on=icd_column, how="left")
            .with_columns(pl.col(axis_name).fill_null("UNCLASSIFIED"))
        )
    return df.with_columns(pl.lit("UNCLASSIFIED").alias(axis_name))


_TRUTHY = ("true", "1", "t", "yes", "y", "sim")
_FALSY = ("false", "0", "f", "no", "n", "nao", "não")


def _restrict_conditions_expr(conditions: list[dict], columns: Iterable[str]) -> pl.Expr | None | bool:
    """Build the AND-list predicate expression (MSD §2.6/§3.10.4 σ restriction).

    Returns the combined `pl.Expr`, or ``False`` when a referenced column is absent
    (the predicate cannot be satisfied → the caller must yield an empty frame), or
    ``None`` when no condition contributed a term (no-op filter)."""
    present = set(columns)
    expr: pl.Expr | None = None
    for cond in conditions:
        column = cond.get("column")
        op = str(cond.get("op") or "")
        value = cond.get("value")
        if not column or str(column) not in present:
            return False
        col = pl.col(str(column))
        if op in {"lt", "le", "gt", "ge"}:
            numeric = col.cast(pl.Float64, strict=False)
            threshold = float(value)
            term = {
                "lt": numeric < threshold,
                "le": numeric <= threshold,
                "gt": numeric > threshold,
                "ge": numeric >= threshold,
            }[op]
        elif op == "eq":
            term = col.cast(pl.Utf8) == str(value)
        elif op == "ne":
            term = col.cast(pl.Utf8) != str(value)
        elif op == "in":
            term = col.cast(pl.Utf8).is_in([str(v) for v in (value or [])])
        elif op == "not_in":
            term = ~col.cast(pl.Utf8).is_in([str(v) for v in (value or [])])
        elif op == "starts_with_any":
            prefixes = [str(v).upper().replace(".", "") for v in (value or []) if str(v).strip()]
            normalized = col.cast(pl.Utf8).str.to_uppercase().str.replace_all(r"\.", "")
            term = pl.any_horizontal([normalized.str.starts_with(prefix) for prefix in prefixes]) if prefixes else pl.lit(False)
        elif op == "is_true":
            term = col.cast(pl.Utf8).str.to_lowercase().is_in(list(_TRUTHY))
        elif op == "is_false":
            term = col.cast(pl.Utf8).str.to_lowercase().is_in(list(_FALSY))
        elif op == "is_not_null":
            term = col.is_not_null()
        else:
            continue
        expr = term if expr is None else (expr & term)
    return expr


def _apply_restrict_conditions_lazy(lf: pl.LazyFrame, conditions: list[dict]) -> pl.LazyFrame:
    """Lazy σ restriction: same predicate semantics as `_apply_restrict_conditions`."""
    expr = _restrict_conditions_expr(conditions, _frame_columns(lf))
    if expr is False:
        return lf.clear()
    if expr is None:
        return lf
    return lf.filter(expr.fill_null(False))


def _apply_restrict_conditions(df: pl.DataFrame, conditions: list[dict]) -> pl.DataFrame:
    """Apply a declarative AND-list predicate (MSD §2.6/§3.10.4 σ restriction).

    Conditions come from health/clinical_event_definitions.yaml; this interpreter is the only
    place the predicate is realized, and it is fully general (no per-event/source code).
    A referenced column that is absent means the predicate cannot be satisfied → empty.
    """
    expr = _restrict_conditions_expr(conditions, df.columns)
    if expr is False:
        return df.clear()
    if expr is None:
        return df
    return df.filter(expr.fill_null(False))


__all__ = [
    "_GEO_SCOPE_PREFIXES",
    "_scope_prefixes_from_intent",
    "MaterializationState",
    "FieldNode",
    "EFGResult",
    "bridge_admin_race_group_counts",
    "load_race_bridge_prior",
    "VALUE_COLUMN",
    "METADATA_COLUMNS",
    "ExecutedField",
    "EFGExecutionReport",
    "_as_dict",
    "_first_existing_path",
    "_source_path",
    "_source_column",
    "_declared_axis_column",
    "_frame_columns",
    "_with_year",
    "_with_year_lazy",
    "_geography_aggregation_of",
    "_with_geo",
    "_with_geo_lazy",
    "_support_frame",
    "_support_frame_lazy",
    "_support_keys",
    "_support_keys_from_columns",
    "_support_keys_lazy",
    "_write",
    "_materialized",
    "_scalar_tensor",
    "_add_icd_stratum",
    "_add_icd_stratum_lazy",
    "_icd_stratum_map_df",
    "_TRUTHY",
    "_FALSY",
    "_apply_restrict_conditions",
    "_apply_restrict_conditions_lazy",
    "_restrict_conditions_expr",
]
