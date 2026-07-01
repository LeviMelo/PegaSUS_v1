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
from pegasus.efg.race_bridge import (
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


def _with_year(df: pl.DataFrame, field: FieldNode | None = None) -> pl.DataFrame:
    source = _declared_axis_column(field, "time")
    if source and source in df.columns:
        return df.with_columns(pl.col(source).cast(pl.Int64, strict=False).alias("year"))
    if "year" in df.columns:
        return df.with_columns(pl.col("year").cast(pl.Int64, strict=False).alias("year"))
    return df.with_columns(pl.lit(None, dtype=pl.Int64).alias("year"))


def _geography_aggregation_of(field: Any) -> str | None:
    """IBGE region level to aggregate event geography to (MSD §3.7), else None."""
    support = _as_dict(getattr(field, "support", {}) or {})
    params = _as_dict(support.get("operator_params") or support.get("params") or {})
    for source in (support, params):
        value = source.get("geography_aggregation")
        if value not in (None, "", "municipality", "none"):
            return str(value)
    return None


def _with_geo(df: pl.DataFrame, field: FieldNode | None = None) -> pl.DataFrame:
    source = _declared_axis_column(field, "geography")
    if source and source in df.columns:
        base = df.with_columns(pl.col(source).cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("municipality_cod6"))
    elif "municipality_cod6" in df.columns:
        base = df
    else:
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("municipality_cod6"))
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
            map_df = pl.DataFrame(
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


def _support_frame(df: pl.DataFrame, field: FieldNode | None = None) -> pl.DataFrame:
    return _with_geo(_with_year(df, field), field)


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


def _add_icd_stratum(df: pl.DataFrame, icd_column: str, level: str, axis_name: str) -> pl.DataFrame:
    """Map each record's ICD code to its chapter/block id (MSD §3.11 σ_C restriction).

    Codes that fall outside the registry-backed groups (or are missing/ill-formed) are
    routed to an explicit ``UNCLASSIFIED`` stratum rather than silently dropped, so the
    cause-specific counts partition the event population exactly.
    """
    from pegasus.datasus.icd_groups import block_for_icd, chapter_for_icd, curated_group_for_icd

    classify = {"chapter": chapter_for_icd, "block": block_for_icd, "curated": curated_group_for_icd}.get(level, chapter_for_icd)
    if icd_column not in df.columns:
        return df.with_columns(pl.lit("UNCLASSIFIED").alias(axis_name))
    codes = df.get_column(icd_column).cast(pl.Utf8, strict=False).drop_nulls().unique().to_list()
    mapping: dict[str, str] = {}
    for code in codes:
        group = classify(code)
        mapping[str(code)] = group.id if group is not None else "UNCLASSIFIED"
    if mapping:
        map_df = pl.DataFrame(
            {icd_column: list(mapping.keys()), axis_name: list(mapping.values())},
            schema={icd_column: pl.Utf8, axis_name: pl.Utf8},
        )
        out = (
            df.with_columns(pl.col(icd_column).cast(pl.Utf8, strict=False))
            .join(map_df, on=icd_column, how="left")
            .with_columns(pl.col(axis_name).fill_null("UNCLASSIFIED"))
        )
    else:
        out = df.with_columns(pl.lit("UNCLASSIFIED").alias(axis_name))
    return out


_TRUTHY = ("true", "1", "t", "yes", "y", "sim")
_FALSY = ("false", "0", "f", "no", "n", "nao", "não")


def _apply_restrict_conditions(df: pl.DataFrame, conditions: list[dict]) -> pl.DataFrame:
    """Apply a declarative AND-list predicate (MSD §2.6/§3.10.4 σ restriction).

    Conditions come from health/clinical_event_definitions.yaml; this interpreter is the only
    place the predicate is realized, and it is fully general (no per-event/source code).
    A referenced column that is absent means the predicate cannot be satisfied → empty.
    """
    expr: pl.Expr | None = None
    for cond in conditions:
        column = cond.get("column")
        op = str(cond.get("op") or "")
        value = cond.get("value")
        if not column or str(column) not in df.columns:
            return df.clear()
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
    if expr is not None:
        df = df.filter(expr.fill_null(False))
    return df


def _count_tensor(field: FieldNode, source: Path) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source), field)
    base_keys = _support_keys(df)
    support = _as_dict(field.support)
    conditions = support.get("restrict_conditions")
    if conditions:
        df = _apply_restrict_conditions(df, list(conditions))
    keys = list(base_keys)
    stratify_icd = support.get("stratify_icd")
    if stratify_icd:
        axis_name = str(support.get("icd_axis") or ("icd_chapter" if stratify_icd == "chapter" else "icd_block"))
        icd_column = str(support.get("icd_column") or "")
        df = _add_icd_stratum(df, icd_column, str(stratify_icd), axis_name)
        keys = [*keys, axis_name]
    # General demographic stratification (MSD §3.7.4): group by a canonical axis derived
    # from a source column, mapping raw codes -> canonical categories so the count joins a
    # matching demographic population denominator. Unknown/total categories are dropped.
    stratify_column = support.get("stratify_column")
    if stratify_column:
        from pegasus.registries.demographic_axis import (
            TOTAL,
            UNKNOWN,
            age_group_for_years,
            canonical_categories,
            source_category_map,
        )

        axis_name = str(support.get("stratify_axis") or stratify_column)
        source_system = str(support.get("stratify_source") or "")
        raw_column = str(stratify_column)
        if raw_column in df.columns:
            tmp = "__canonical_stratum__"
            if axis_name == "age_group":
                # Age is a direct arithmetic bucketing of the source's single-year age
                # field (SIM/SINASC/SIH carry age_years / maternal_age_years), NOT a
                # category-code crosswalk (MSD §3.7.4). Bucket to the canonical age_N
                # basis so the count joins the single-year population denominator.
                df = df.with_columns(
                    pl.col(raw_column)
                    .map_elements(age_group_for_years, return_dtype=pl.Utf8)
                    .alias(axis_name)
                ).filter(~pl.col(axis_name).is_in([TOTAL, UNKNOWN]))
                keys = [*keys, axis_name]
            elif axis_name == "race":
                # Administrative race/color is NOT self-declared census race: a direct
                # code->canonical crosswalk here would be silent redistribution
                # (§3.7.4). Keep the RAW admin code as the stratum so Bridge_R can map it
                # downstream; this count never divides a self-declared population directly
                # (align_fields gates race rates on race_bridge_required).
                df = df.with_columns(
                    pl.col(raw_column).cast(pl.Utf8, strict=False).alias(axis_name)
                ).filter(pl.col(axis_name).is_not_null())
                keys = [*keys, axis_name]
            else:
                # sex (and any future direct-crosswalk axis). Accept BOTH raw source codes
                # and already-canonical values: the normalizers may emit the canonical
                # category directly (SIM/SINASC/SIH 'sex' is 'male'/'female', not '1'/'2'),
                # so identity on the canonical vocabulary keeps a stale/no-op code map from
                # dropping every row into __unknown__.
                code_map = source_category_map(axis_name, source_system)
                mapping = {**{c: c for c in canonical_categories(axis_name)}, **code_map}
                if mapping:
                    map_df = pl.DataFrame(
                        {raw_column: list(mapping.keys()), tmp: list(mapping.values())},
                        schema={raw_column: pl.Utf8, tmp: pl.Utf8},
                    )
                    df = (
                        df.with_columns(pl.col(raw_column).cast(pl.Utf8, strict=False))
                        .join(map_df, on=raw_column, how="left")
                        .with_columns(pl.col(tmp).fill_null(UNKNOWN).alias(axis_name))
                        .filter(~pl.col(axis_name).is_in([TOTAL, UNKNOWN]))
                        .drop(tmp)
                    )
                    keys = [*keys, axis_name]
    if keys:
        out = df.group_by(keys).agg(pl.len().cast(pl.Float64).alias(VALUE_COLUMN)).sort(keys)
    else:
        out = pl.DataFrame({VALUE_COLUMN: [float(df.height)]})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "count_measure").alias("operator"),
    ])
    return out


def _functional_tensor(field: FieldNode, source: Path) -> pl.DataFrame:
    """Materialize a statistical functional (mean/median) of a per-record mark over each
    support cell (MSD §3.10.4-6 Ψ operators)."""
    df = _support_frame(pl.read_parquet(source), field)
    support = _as_dict(field.support)
    mark = str(support.get("mark_column") or "")
    functional = str(support.get("functional") or "mean")
    if mark not in df.columns:
        raise ValueError(f"functional field {field.id} mark column {mark!r} absent from source")
    df = df.with_columns(pl.col(mark).cast(pl.Float64, strict=False).alias("__mark__"))
    agg = pl.col("__mark__").median() if functional == "median" else pl.col("__mark__").mean()
    keys = _support_keys(df)
    if keys:
        out = df.group_by(keys).agg(agg.cast(pl.Float64).alias(VALUE_COLUMN)).sort(keys)
    else:
        scalar = df.get_column("__mark__").median() if functional == "median" else df.get_column("__mark__").mean()
        out = pl.DataFrame({VALUE_COLUMN: [float(scalar) if scalar is not None else None]})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(f"psi_{functional}").alias("operator"),
    ])
    return out


def _sum_tensor(field: FieldNode, source: Path, column: str) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source), field)
    df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias("__value__"))
    keys = _support_keys(df)
    if keys:
        out = df.group_by(keys).agg(pl.col("__value__").sum().cast(pl.Float64).alias(VALUE_COLUMN)).sort(keys)
    else:
        out = pl.DataFrame({VALUE_COLUMN: [float(df["__value__"].sum() or 0.0)]})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "sigma_C").alias("operator"),
    ])
    return out


def _source_field_tensor(field: FieldNode, source: Path, column: str) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source), field)
    keys = _support_keys(df)
    out = df.select([
        *(pl.col(key) for key in keys),
        pl.col(column).alias(VALUE_COLUMN),
    ]).with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "source_field").alias("operator"),
    ])
    return out


def _load_parent_tensor(parent: FieldNode) -> pl.DataFrame:
    if not parent.path:
        raise ValueError(f"parent {parent.id} has no materialized path")
    path = Path(parent.path)
    if not path.exists():
        raise FileNotFoundError(f"parent tensor missing: {path}")
    df = pl.read_parquet(path)
    if VALUE_COLUMN not in df.columns:
        raise ValueError(f"parent tensor lacks {VALUE_COLUMN!r}: {path}")
    return df


_YEAR_KEYS = ("year", "admission_year", "birth_year", "competence_year")


def _temporal_lag_of(field: Any) -> int:
    """Year lag declared for a bridge field (support/operator_params), else 0."""
    support = _as_dict(getattr(field, "support", {}) or {})
    params = _as_dict(support.get("operator_params") or support.get("params") or {})
    for source in (support, params):
        value = source.get("temporal_lag")
        if value is not None:
            try:
                k = int(value)
            except (TypeError, ValueError):
                return 0
            return k if k > 0 else 0
    return 0


def _shift_year(df: pl.DataFrame, lag: int) -> pl.DataFrame:
    """Add `lag` to the first present year column so left(t) aligns to right(t+lag).

    Output cell year t then pairs the left operand's value from year t-lag (MSD §2.11
    delayed cross-source effect). Pure and column-agnostic across the SIH/SINASC/SIM
    year axis names."""
    if lag <= 0:
        return df
    for column in _YEAR_KEYS:
        if column in df.columns:
            return df.with_columns((pl.col(column).cast(pl.Int64, strict=False) + lag).alias(column))
    return df


def _join_keys(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    common = [column for column in left.columns if column in right.columns]
    return [column for column in common if column not in METADATA_COLUMNS and column != VALUE_COLUMN]


def _compute_rn_ratio(field: FieldNode, parents_by_id: dict[str, FieldNode], output_dir: Path) -> tuple[Path, int, dict[str, Any]]:
    parent_ids = list(field.lineage.parent_ids or [])
    if len(parent_ids) < 2:
        raise ValueError(f"RN field {field.id} requires numerator and denominator parents")
    numerator = parents_by_id[parent_ids[0]]
    denominator = parents_by_id[parent_ids[1]]
    n = _load_parent_tensor(numerator).rename({VALUE_COLUMN: "value_numerator"})
    d = _load_parent_tensor(denominator).rename({VALUE_COLUMN: "value_denominator"})
    keys = _join_keys(n, d)
    
    # Ecological Fallacy Guard: Aggregate numerator up to denominator's spatial support if mismatched.
    if "municipality_cod6" in n.columns and "municipality_cod6" not in d.columns:
        agg_keys = [k for k in keys if k != "municipality_cod6"]
        if agg_keys:
            n = n.group_by(agg_keys).agg(pl.col("value_numerator").sum())
        else:
            n = pl.DataFrame({"value_numerator": [n["value_numerator"].sum()]})
        keys = _join_keys(n, d)

    if not keys:
        raise RuntimeError(
            "RN operator requires at least one intersecting support axis; "
            "cross-join is forbidden to prevent OOM and indicates failed Δ support alignment."
        )
    # Stratifier columns present on the numerator but absent from the (unstratified)
    # denominator — e.g. icd_chapter/icd_block from a σ_C restriction. They are NOT join
    # keys (the denominator broadcasts across strata) but MUST survive into the output,
    # otherwise cause-specific rates collapse to one ambiguous row per (year, municipality).
    numerator_strata = [
        column
        for column in n.columns
        if column not in keys
        and column != "value_numerator"
        and column not in METADATA_COLUMNS
    ]
    joined = n.join(d, on=keys, how="left", suffix="_denominator")

    missing_denom_count = joined.filter(pl.col("value_denominator").is_null() | pl.col("value_denominator").is_nan()).height
    denom_fragility = float(missing_denom_count) / float(joined.height) if joined.height > 0 else 1.0

    parent_denom_id = parent_ids[1]
    parent_fragility = float(parents_by_id[parent_denom_id].support.get("denom_fragility", 0.0))
    combined_fragility = min(1.0, parent_fragility + denom_fragility)

    out = joined.with_columns(
        pl.when((pl.col("value_denominator") > 0) & pl.col("value_denominator").is_not_null())
        .then(pl.col("value_numerator") / pl.col("value_denominator"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias(VALUE_COLUMN)
    )

    keep = [column for column in [*keys, *numerator_strata] if column in out.columns]
    out = out.select([
        *[pl.col(column) for column in keep],
        pl.col(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("RN").alias("operator"),
    ])
    path, rows = _write(output_dir / f"{field.id}.parquet", out)
    # Report the realized numerator/denominator totals so the Q-tensor (MSD §3.12)
    # carries the true event/denominator counts for this ratio rather than 0/None.
    num_total = float(n.select(pl.col("value_numerator").sum()).item() or 0.0)
    den_total = float(d.select(pl.col("value_denominator").sum()).item() or 0.0)
    return path, rows, {
        "denom_fragility": combined_fragility,
        "n_events": num_total,
        "n_denom": den_total if den_total > 0 else None,
    }



def _is_bridge_divergence(field: FieldNode) -> bool:
    op = str(field.operator or "").lower()
    kind = str(field.kind or "").lower()
    support = _as_dict(field.support)
    params = _as_dict(support.get("operator_params") or support.get("params") or {})
    haystack = " ".join([
        op,
        kind,
        str(support.get("bridge_type") or "").lower(),
        str(params.get("bridge_type") or "").lower(),
        str(params.get("operator") or "").lower(),
    ])
    return (
        "divergence" in haystack
        or "morbidity_mortality" in haystack
        or "mortality_morbidity" in haystack
        or op in {"bridge_divergence", "divergence_log_ratio"}
    )


def _is_race_bridge(field: FieldNode) -> bool:
    support = _as_dict(field.support)
    params = _as_dict(field.lineage.operator_params)
    return (
        field.operator == "Bridge_R_fixedC_dynamic_weight"
        or field.operator == "Bridge_R_localPi_posteriorC"
        or support.get("bridge_operator") == "Bridge_R_fixedC_dynamic_weight"
        or support.get("bridge_operator") == "Bridge_R_localPi_posteriorC"
        or params.get("bridge_operator") == "Bridge_R_fixedC_dynamic_weight"
        or params.get("bridge_operator") == "Bridge_R_localPi_posteriorC"
    )


def _race_column(field: FieldNode, parent: FieldNode, df: pl.DataFrame) -> str:
    support = _as_dict(parent.support)
    candidates = [
        support.get("column"),
        support.get("source_column"),
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) in df.columns:
            return str(candidate)
    raise ValueError(
        f"Bridge_R field {field.id} parent {parent.id} lacks a registry-declared "
        "administrative race source column present in the source artifact"
    )


def _fixedc_support_groups(df: pl.DataFrame) -> list[tuple[dict[str, Any], pl.DataFrame]]:
    keys = _support_keys(df)
    if not keys:
        return [({}, df)]
    rows = df.select(keys).unique().sort(keys).iter_rows(named=True)
    groups: list[tuple[dict[str, Any], pl.DataFrame]] = []
    for values in rows:
        sub = df
        for key, value in values.items():
            if value is None:
                sub = sub.filter(pl.col(key).is_null())
            else:
                sub = sub.filter(pl.col(key) == value)
        groups.append((dict(values), sub))
    return groups


def _compute_race_bridge_tensor(
    field: FieldNode,
    parent: FieldNode,
    output_dir: Path,
) -> tuple[Path, int, dict[str, Any]]:
    params = {**_as_dict(field.support), **_as_dict(field.lineage.operator_params)}
    prior_path = params.get("prior_path") or params.get("bridge_prior_path")
    if not prior_path:
        raise ValueError(f"Bridge_R field {field.id} missing prior_path")
    prior = load_race_bridge_prior(prior_path)

    source = _source_path(parent)
    if source is None:
        raise ValueError(f"Bridge_R parent {parent.id} has no source artifact path")
    df = _support_frame(pl.read_parquet(source), parent)
    race_column = _race_column(field, parent, df)
    state_col = "race_missingness_state" if "race_missingness_state" in df.columns else None

    rows: list[dict[str, Any]] = []
    summary_missing = 0
    summary_total = 0
    summary_cv: list[float] = []
    summary_width: list[float] = []
    for support_values, group in _fixedc_support_groups(df):
        race_values = group[race_column].to_list()
        states = group[state_col].to_list() if state_col is not None else None
        posterior = bridge_admin_race_group_counts(
            race_codes=race_values,
            race_states=states,
            prior=prior,
            support={**support_values, "n_events": int(group.height)},
        )
        raw_counts = posterior.raw_admin_counts
        metadata = posterior.metadata()
        summary_missing += posterior.missing_count
        summary_total += int(group.height)
        summary_cv.append(float(posterior.race_bridge_cv))
        summary_width.append(float(posterior.sensitivity_width))
        for target in prior.target_categories:
            rows.append({
                **support_values,
                "target_race_category": target,
                VALUE_COLUMN: float(posterior.posterior_counts[target]),
                "lower_count": float(posterior.lower_counts[target]),
                "upper_count": float(posterior.upper_counts[target]),
                "raw_admin_counts_json": json.dumps(raw_counts, sort_keys=True),
                "missing_count": int(posterior.missing_count),
                "missing_race_share": float(posterior.missing_share),
                "race_bridge_cv": float(posterior.race_bridge_cv),
                "sensitivity_width": float(posterior.sensitivity_width),
                "prior_hash": prior.prior_hash,
                "bridge_mode": prior.mode,
                "bridge_operator": "Bridge_R_localPi_posteriorC",
                "field_id": field.id,
                "field_name": field.name,
                "operator": "Bridge_R_localPi_posteriorC",
                "bridge_metadata_json": json.dumps(metadata, sort_keys=True, default=str),
            })

    out = pl.DataFrame(rows) if rows else pl.DataFrame({
        "field_id": [field.id],
        "field_name": [field.name],
        "operator": ["Bridge_R_localPi_posteriorC"],
        VALUE_COLUMN: [0.0],
        "lower_count": [0.0],
        "upper_count": [0.0],
        "missing_race_share": [0.0],
        "race_bridge_cv": [0.0],
        "sensitivity_width": [prior.sensitivity_width],
        "prior_hash": [prior.prior_hash],
        "bridge_mode": [prior.mode],
        "bridge_operator": ["Bridge_R_localPi_posteriorC"],
    })
    path, row_count = _write(output_dir / f"{field.id}.parquet", out)
    metadata = {
        "missing_race_share": (summary_missing / float(summary_total)) if summary_total else 0.0,
        "race_bridge_cv": max(summary_cv) if summary_cv else 0.0,
        "sensitivity_width": max(summary_width) if summary_width else prior.sensitivity_width,
        "prior_hash": prior.prior_hash,
        "bridge_mode": prior.mode,
        "bridge_operator": "Bridge_R_localPi_posteriorC",
        "raw_admin_counts_preserved": True,
        "missing_category_preserved": True,
    }
    return path, row_count, metadata


def _compute_bridge_tensor(field: FieldNode, parents_by_id: dict[str, FieldNode], output_dir: Path) -> tuple[Path, int, dict[str, Any] | None]:
    parent_ids = list(field.lineage.parent_ids or [])
    if not parent_ids:
        out = pl.DataFrame({
            "field_id": [field.id],
            "field_name": [field.name],
            "operator": [field.operator or "bridge"],
            VALUE_COLUMN: [None],
        }).cast({VALUE_COLUMN: pl.Float64})
        path, rows = _write(output_dir / f"{field.id}.parquet", out)
        return path, rows, None

    if _is_race_bridge(field):
        return _compute_race_bridge_tensor(field, parents_by_id[parent_ids[0]], output_dir)

    if _is_bridge_divergence(field) and len(parent_ids) == 2:
        p0 = _load_parent_tensor(parents_by_id[parent_ids[0]])
        p1 = _load_parent_tensor(parents_by_id[parent_ids[1]])
        # Temporal-lag divergence (MSD §2.11): shift the left operand's year by k so
        # the output cell at year t pairs left(t-k) with right(t) — log(left(t-k)/right(t)).
        # Declared by the bridge grammar; 0 = the standard contemporaneous divergence.
        lag = _temporal_lag_of(field)
        if lag:
            p0 = _shift_year(p0, lag)
        keys = _join_keys(p0, p1)
        if not keys:
            raise RuntimeError("Bridge divergence requires intersecting support axes.")

        joined = p0.join(p1, on=keys, how="inner", suffix="_right")
        epsilon = 1e-9
        out = joined.with_columns(
            ((pl.col(VALUE_COLUMN) + epsilon) / (pl.col(VALUE_COLUMN + "_right") + epsilon))
            .log()
            .cast(pl.Float64)
            .alias(VALUE_COLUMN)
        )
        keep = [c for c in keys if c in out.columns]
        out = out.select([
            *[pl.col(c) for c in keep],
            pl.col(VALUE_COLUMN),
            pl.lit(field.id).alias("field_id"),
            pl.lit(field.name).alias("field_name"),
            pl.lit(field.operator or "divergence_log_ratio").alias("operator"),
        ])
        path, rows = _write(output_dir / f"{field.id}.parquet", out)
        return path, rows, None

    # Fallback for Bridge_R / unary bridges
    if str(field.operator or "").startswith("Bridge_R"):
        raise ValueError(f"Bridge_R operator {field.operator!r} has no physical executor")
    parent = parents_by_id[parent_ids[0]]
    df = _load_parent_tensor(parent)
    out = df.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "Bridge_R").alias("operator"),
    ])
    path, rows = _write(output_dir / f"{field.id}.parquet", out)
    return path, rows, None


def _sidra_population_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize the SIDRA population denominator as a per-municipality panel
    (municipality_cod6, year, value) so the Radon-Nikodym rate join matches each
    municipality's deaths/births to its own population — the national grid."""
    support = _as_dict(field.support)
    facts_path = support.get("sidra_facts_path") or support.get("artifact_path")
    if not facts_path:
        raise ValueError("SIDRA population anchor field has no sidra_facts_path")
    from pegasus.sidra.population_cube.anchor import load_sidra_population_totals_frame

    frame = load_sidra_population_totals_frame(facts_path)
    if frame.height == 0:
        raise ValueError("SIDRA population facts contain no total-category population rows")
    out = frame.with_columns(
        pl.col("value").cast(pl.Float64).alias(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("sidra_population_total_anchor").alias("operator"),
    )
    return _write(output_dir / f"{field.id}.parquet", out)


def _sidra_context_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize a SIDRA context gradient (§3.10.7 V_X) as a per-municipality panel
    (year, municipality_cod6, value). locality_id is cod7; municipality_cod6 = cod7[:6],
    matching the DATASUS event geography so PIRS can use it as a covariate."""
    support = _as_dict(field.support)
    facts_path = support.get("sidra_facts_path") or support.get("artifact_path")
    if not facts_path:
        raise ValueError("SIDRA context field has no facts path")
    table_id = str(support.get("table_id") or "")
    variable_id = str(support.get("variable_id") or "")
    df = pl.read_parquet(facts_path)
    if "table_id" in df.columns and table_id:
        df = df.filter(pl.col("table_id").cast(pl.Utf8) == table_id)
    if "variable_id" in df.columns and variable_id:
        df = df.filter(pl.col("variable_id").cast(pl.Utf8) == variable_id)
    value_col = "value_numeric" if "value_numeric" in df.columns else VALUE_COLUMN
    out = df.with_columns([
        pl.col("locality_id").cast(pl.Utf8).str.slice(0, 6).alias("municipality_cod6"),
        pl.col("period").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64, strict=False).alias("year"),
    ]).select([
        pl.col("year"),
        pl.col("municipality_cod6"),
        pl.col(value_col).cast(pl.Float64, strict=False).alias(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("sidra_context_field").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


def _category_code_for_classification(category_tuple_raw: Any, classification_id: str) -> str | None:
    import json
    try:
        pairs = json.loads(category_tuple_raw) if isinstance(category_tuple_raw, str) else category_tuple_raw
    except Exception:
        return None
    for pair in pairs or []:
        if pair and str(pair[0]) == str(classification_id):
            return str(pair[1])
    return None


def _sidra_demographic_population_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize a demographic-stratified population panel (MSD §2.8):
    (year, municipality_cod6, <axis>, value), axis category canonicalized via the
    demographic-axis registry, dropping the marginal Total and unknown categories."""
    from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, map_category

    support = _as_dict(field.support)
    facts_path = support.get("sidra_facts_path") or support.get("artifact_path")
    classification_id = str(support.get("classification_id") or "")
    axis = str(support.get("demographic_axis") or "stratum")
    if not facts_path:
        raise ValueError("demographic population field has no facts path")
    df = pl.read_parquet(facts_path)
    value_col = "value_numeric" if "value_numeric" in df.columns else VALUE_COLUMN
    rows: list[dict[str, Any]] = []
    for record in df.iter_rows(named=True):
        code = _category_code_for_classification(record.get("category_tuple"), classification_id)
        canonical = map_category(axis, "SIDRA", code) if code is not None else UNKNOWN
        if canonical in {TOTAL, UNKNOWN}:
            continue  # marginal/unknown is not a stratum of the disaggregated tensor
        locality = str(record.get("locality_id") or "")
        period = str(record.get("period") or "")
        value = record.get(value_col)
        rows.append({
            "year": int(period[:4]) if period[:4].isdigit() else None,
            "municipality_cod6": locality[:6],
            axis: canonical,
            VALUE_COLUMN: float(value) if value is not None else None,
        })
    out = pl.DataFrame(rows) if rows else pl.DataFrame({VALUE_COLUMN: []}, schema={VALUE_COLUMN: pl.Float64})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("sidra_demographic_population").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


def _population_solver_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize a solver-produced population tensor as an EFG denominator panel."""
    support = _as_dict(field.support)
    tensor_path = support.get("population_tensor_path") or support.get("artifact_path")
    if not tensor_path:
        raise ValueError("population solver field has no population_tensor_path")
    df = pl.read_parquet(tensor_path)
    required = {"year", "municipality_cod6", VALUE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"population solver tensor missing columns: {sorted(missing)}")
    # This field is the MARGINAL population over a single demographic axis (or the
    # crude total when None): sum the full (age,sex,race) tensor over every axis
    # except this field's, so a sex-stratified numerator divides by the sex-marginal
    # population (MSD §3.7.4). `marginal_demographic_axis` is set by the materializer.
    marginal_axis = support.get("marginal_demographic_axis")
    keep_axes = [marginal_axis] if (marginal_axis and marginal_axis in df.columns) else []
    group_keys = ["year", "municipality_cod6", *keep_axes]
    agg = (
        df.with_columns(
            pl.col("year").cast(pl.Int64, strict=False),
            pl.col("municipality_cod6").cast(pl.Utf8),
            *[pl.col(axis).cast(pl.Utf8) for axis in keep_axes],
            pl.col(VALUE_COLUMN).cast(pl.Float64, strict=False),
        )
        .group_by(group_keys)
        .agg(pl.col(VALUE_COLUMN).sum().alias(VALUE_COLUMN))
    )
    out = agg.select([
        *group_keys,
        pl.col(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("population_tensor_solver").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


def _execute_non_rn(field: FieldNode, output_dir: Path, intent: Any = None) -> tuple[Path, int]:
    op = str(field.operator or "").lower()
    source = _source_path(field)

    # Only genuine pre-computed scalar anchors (e.g. the SIDRA population total)
    # use the scalar path. Count/aggregation fields carry incidental scalar keys
    # (n_events, count) in their support and MUST NOT short-circuit to a single
    # global value — they have to group by (year, municipality) over the source,
    # otherwise the Radon-Nikodym rate join finds no shared support axis.
    use_scalar = "anchor" in op or source is None
    if use_scalar:
        scalar = _scalar_tensor(field, output_dir)
        if scalar is not None:
            return scalar

    if source is None:
        raise ValueError("no source parquet path and no scalar support value")
    head = pl.read_parquet(source, n_rows=25)
    column = _source_column(field, head)

    if op in {"count_measure", "count", "event_count"} or field.unit in {"counts", "count"}:
        out = _count_tensor(field, source)
    elif column is not None and field.aggregation in {"additive", "statistical_functional", "weighted_mean"}:
        out = _sum_tensor(field, source, column)
    elif column is not None:
        out = _source_field_tensor(field, source, column)
    else:
        out = _count_tensor(field, source)

    geo_mode = "native"
    if intent is not None:
        geo_mode = getattr(intent, "geo_mode", "native")
        if isinstance(intent, dict):
            geo_mode = intent.get("geo_mode", geo_mode)
            
    if geo_mode == "AMC" and "municipality_cod6" in out.columns:
        from pegasus.geo.amc import contract_to_amc
        # Look for the AMC crosswalk relative to the data lake root
        crosswalk_path = Path("data/raw/geo/amc_crosswalk.parquet")
        if crosswalk_path.exists():
            try:
                amc_result = contract_to_amc(out, crosswalk_path=str(crosswalk_path), value_column=VALUE_COLUMN, municipality_column="municipality_cod6")
                out = amc_result.frame.rename({"amc_id": "municipality_cod6"})
            except Exception:
                pass # Fallback to native if crosswalk fails

    return _write(output_dir / f"{field.id}.parquet", out)


def execute_efg_result(
    efg: EFGResult,
    *,
    output_dir: str | Path,
    require_materialized: bool = True,
    intent: Any = None,
) -> tuple[EFGResult, EFGExecutionReport]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scope_token = _GEO_SCOPE_PREFIXES.set(_scope_prefixes_from_intent(intent))
    try:
        return _execute_efg_result_impl(efg, out_dir=out_dir, require_materialized=require_materialized, intent=intent)
    finally:
        _GEO_SCOPE_PREFIXES.reset(scope_token)


def _execute_efg_result_impl(
    efg: EFGResult,
    *,
    out_dir: Path,
    require_materialized: bool,
    intent: Any = None,
) -> tuple[EFGResult, EFGExecutionReport]:
    fields_by_id: dict[str, FieldNode] = {field.id: field for field in efg.fields}
    executed: list[ExecutedField] = []

    # Fixed-point execution: source/scalar fields first, then RN and bridges.
    pending = set(fields_by_id)
    last_error: dict[str, str] = {}
    for _ in range(max(2, len(fields_by_id) + 1)):
        progressed = False
        for field_id in list(pending):
            field = fields_by_id[field_id]
            op = str(field.operator or "")
            try:
                if op.upper() == "RN" or field.kind == "intensive_density":
                    path, rows, support_update = _compute_rn_ratio(field, fields_by_id, out_dir)
                elif op == "sidra_population_total_anchor":
                    path, rows = _sidra_population_tensor(field, out_dir)
                    support_update = None
                elif op == "sidra_context_field":
                    path, rows = _sidra_context_tensor(field, out_dir)
                    support_update = None
                elif op == "sidra_demographic_population":
                    path, rows = _sidra_demographic_population_tensor(field, out_dir)
                    support_update = None
                elif op == "psi_functional":
                    functional_source = _source_path(field)
                    if functional_source is None:
                        raise ValueError(f"functional field {field.id} has no source artifact")
                    path, rows = _write(out_dir / f"{field.id}.parquet", _functional_tensor(field, functional_source))
                    support_update = None
                elif op == "population_tensor_solver":
                    path, rows = _population_solver_tensor(field, out_dir)
                    support_update = None
                elif op.startswith("Bridge") or "bridge" in op.lower() or field.kind in {"bridge_module", "bridge_divergence"}:
                    path, rows, support_update = _compute_bridge_tensor(field, fields_by_id, out_dir)
                else:
                    path, rows = _execute_non_rn(field, out_dir, intent)
                    support_update = None
                if support_update:
                    field = field.model_copy(update={
                        "support": {**dict(field.support), **support_update},
                    })
                new_field = _materialized(field, path)
                fields_by_id[field_id] = new_field
                executed.append(ExecutedField(new_field, "success", str(path), rows))
                pending.remove(field_id)
                progressed = True
            except Exception as exc:
                # RN may be waiting for parent tensors. Keep it pending until the next pass,
                # but remember the real error so a permanently-blocked field reports WHY
                # instead of a generic "parents_not_materialized".
                last_error[field_id] = f"{type(exc).__name__}: {exc}"
                # RN and cross-source bridges/divergences depend on parent tensors that may
                # not be materialized yet; keep them pending across fixed-point passes.
                if (
                    op.upper() == "RN"
                    or field.kind in {"intensive_density", "bridge_divergence", "bridge_module"}
                    or op.startswith("Bridge")
                    or op == "divergence_log_ratio"
                ):
                    continue
                executed.append(ExecutedField(field, "blocked", None, 0, last_error[field_id]))
                pending.remove(field_id)
                progressed = True
        if not pending or not progressed:
            break

    for field_id in sorted(pending):
        field = fields_by_id[field_id]
        reason = last_error.get(field_id, "parents_not_materialized_or_operator_not_executable")
        executed.append(ExecutedField(field, "blocked", None, 0, reason))

    blocked = [item for item in executed if item.status != "success"]
    if require_materialized and blocked:
        msg = "; ".join(f"{item.field.name}:{item.reason}" for item in blocked[:40])
        raise RuntimeError(f"Autonomous EFG physical execution blocked {len(blocked)} fields: {msg}")

    new_efg = EFGResult(
        schema_version=efg.schema_version,
        efg_id=efg.efg_id,
        substrate_id=efg.substrate_id,
        fields=tuple(fields_by_id[field.id] for field in efg.fields),
        edges=efg.edges,
        failed_branches=efg.failed_branches,
        warnings=efg.warnings,
        variable_dictionary=efg.variable_dictionary,
        precompression=efg.precompression,
        source_hashes=efg.source_hashes,
        registry_hashes=efg.registry_hashes,
        legality_summary=efg.legality_summary,
        operator_mode=efg.operator_mode,
        core_seed_summary=efg.core_seed_summary,
        bridge_plan_summary=efg.bridge_plan_summary,
        domain_summaries=efg.domain_summaries,
    )
    report = EFGExecutionReport(
        status="success" if not blocked else "blocked",
        executed_count=sum(1 for item in executed if item.status == "success"),
        blocked_count=len(blocked),
        fields=tuple(executed),
        output_dir=str(out_dir),
    )
    return new_efg, report


__all__ = ["execute_efg_result", "EFGExecutionReport", "ExecutedField"]
