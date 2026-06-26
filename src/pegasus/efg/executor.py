"""Physical tensor executor for autonomous EFG.

Operators here execute arrays. Metadata-only operators are not sufficient for
MSD convergence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from pegasus.core.enums import MaterializationState
from pegasus.core.schemas import FieldNode
from pegasus.efg.dag import EFGResult
from pegasus.efg.race_bridge import (
    ADMIN_RACE_LABELS,
    RaceBridgeCounts,
    fixedc_dynamic_weight_bridge,
    load_race_bridge_prior,
)


VALUE_COLUMN = "value"

GEO_COLUMNS = (
    "municipality_cod6",
    "mun_residence_cod6",
    "mun_movement_cod6",
    "mun_occurrence_cod6",
    "CODMUNRES",
    "CODMUNOCOR",
    "MUNIC_RES",
    "MUNIC_MOV",
)

YEAR_COLUMNS = (
    "year",
    "death_year",
    "birth_year",
    "admission_year",
    "period_year",
    "ANO",
)

DATE_COLUMNS = (
    "death_date",
    "birth_date",
    "admit_date",
    "event_date",
    "DTOBITO",
    "DTNASC",
    "DT_INTER",
)

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


def _with_year(df: pl.DataFrame) -> pl.DataFrame:
    if "year" in df.columns:
        return df.with_columns(pl.col("year").cast(pl.Int64, strict=False).alias("year"))
    for column in YEAR_COLUMNS:
        if column in df.columns:
            return df.with_columns(pl.col(column).cast(pl.Int64, strict=False).alias("year"))
    for column in DATE_COLUMNS:
        if column in df.columns:
            return df.with_columns(pl.col(column).cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64, strict=False).alias("year"))
    return df.with_columns(pl.lit(None, dtype=pl.Int64).alias("year"))


def _with_geo(df: pl.DataFrame) -> pl.DataFrame:
    if "municipality_cod6" in df.columns:
        base = df
    else:
        source = next((column for column in GEO_COLUMNS if column in df.columns), None)
        if source is None:
            return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("municipality_cod6"))
        base = df.with_columns(pl.col(source).cast(pl.Utf8).str.extract(r"(\d{6})", 1).alias("municipality_cod6"))
    return base.filter(~pl.col("municipality_cod6").cast(pl.Utf8).str.contains(r"^\d{2}0000$").fill_null(False))


def _support_frame(df: pl.DataFrame) -> pl.DataFrame:
    return _with_geo(_with_year(df))


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


def _count_tensor(field: FieldNode, source: Path) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source))
    keys = _support_keys(df)
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


def _sum_tensor(field: FieldNode, source: Path, column: str) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source))
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
    df = _support_frame(pl.read_parquet(source))
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

    keep = [column for column in keys if column in out.columns]
    out = out.select([
        *[pl.col(column) for column in keep],
        pl.col(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("RN").alias("operator"),
    ])
    path, rows = _write(output_dir / f"{field.id}.parquet", out)
    return path, rows, {"denom_fragility": combined_fragility}



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


def _is_fixedc_race_bridge(field: FieldNode) -> bool:
    support = _as_dict(field.support)
    params = _as_dict(field.lineage.operator_params)
    return (
        field.operator == "Bridge_R_fixedC_dynamic_weight"
        or support.get("bridge_operator") == "Bridge_R_fixedC_dynamic_weight"
        or params.get("bridge_operator") == "Bridge_R_fixedC_dynamic_weight"
    )


def _admin_code(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or None


def _race_column(field: FieldNode, parent: FieldNode, df: pl.DataFrame) -> str:
    support = _as_dict(parent.support)
    candidates = [
        support.get("column"),
        support.get("source_column"),
        "race_color_admin",
        "RACACOR",
        "RACA_COR",
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) in df.columns:
            return str(candidate)
    raise ValueError(f"Bridge_R field {field.id} could not locate an administrative race column")


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


def _compute_fixedc_race_bridge_tensor(
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
    df = _support_frame(pl.read_parquet(source))
    race_column = _race_column(field, parent, df)
    state_col = "race_missingness_state" if "race_missingness_state" in df.columns else None

    rows: list[dict[str, Any]] = []
    summary_missing = 0
    summary_total = 0
    summary_cv: list[float] = []
    summary_width: list[float] = []
    for support_values, group in _fixedc_support_groups(df):
        raw_counts = {code: 0 for code in ADMIN_RACE_LABELS}
        missing = 0
        race_values = group[race_column].to_list()
        states = group[state_col].to_list() if state_col is not None else [None] * len(race_values)
        for code_value, state in zip(race_values, states, strict=False):
            code = _admin_code(code_value)
            if code in raw_counts and (state in (None, "valid_admin_race")):
                raw_counts[code] += 1
            else:
                missing += 1
        counts = RaceBridgeCounts(
            raw_admin_counts=raw_counts,
            missing_count=missing,
            total_count=int(group.height),
            support={**support_values, "n_events": int(group.height)},
        )
        posterior = fixedc_dynamic_weight_bridge(counts, prior)
        metadata = posterior.metadata()
        summary_missing += missing
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
                "missing_count": int(missing),
                "missing_race_share": float(posterior.missing_share),
                "race_bridge_cv": float(posterior.race_bridge_cv),
                "sensitivity_width": float(posterior.sensitivity_width),
                "prior_hash": prior.prior_hash,
                "bridge_mode": prior.mode,
                "bridge_operator": "Bridge_R_fixedC_dynamic_weight",
                "field_id": field.id,
                "field_name": field.name,
                "operator": "Bridge_R_fixedC_dynamic_weight",
                "bridge_metadata_json": json.dumps(metadata, sort_keys=True, default=str),
            })

    out = pl.DataFrame(rows) if rows else pl.DataFrame({
        "field_id": [field.id],
        "field_name": [field.name],
        "operator": ["Bridge_R_fixedC_dynamic_weight"],
        VALUE_COLUMN: [0.0],
        "lower_count": [0.0],
        "upper_count": [0.0],
        "missing_race_share": [0.0],
        "race_bridge_cv": [0.0],
        "sensitivity_width": [prior.sensitivity_width],
        "prior_hash": [prior.prior_hash],
        "bridge_mode": [prior.mode],
        "bridge_operator": ["Bridge_R_fixedC_dynamic_weight"],
    })
    path, row_count = _write(output_dir / f"{field.id}.parquet", out)
    metadata = {
        "missing_race_share": (summary_missing / float(summary_total)) if summary_total else 0.0,
        "race_bridge_cv": max(summary_cv) if summary_cv else 0.0,
        "sensitivity_width": max(summary_width) if summary_width else prior.sensitivity_width,
        "prior_hash": prior.prior_hash,
        "bridge_mode": prior.mode,
        "bridge_operator": "Bridge_R_fixedC_dynamic_weight",
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

    if _is_fixedc_race_bridge(field):
        return _compute_fixedc_race_bridge_tensor(field, parents_by_id[parent_ids[0]], output_dir)

    if _is_bridge_divergence(field) and len(parent_ids) == 2:
        p0 = _load_parent_tensor(parents_by_id[parent_ids[0]])
        p1 = _load_parent_tensor(parents_by_id[parent_ids[1]])
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
    from pegasus.she.population.sidra_anchor import load_sidra_population_totals_frame

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
                elif op.startswith("Bridge") or "bridge" in op.lower() or field.kind in {"bridge_module", "bridge_divergence"}:
                    path, rows, support_update = _compute_bridge_tensor(field, fields_by_id, out_dir)
                else:
                    path, rows = _execute_non_rn(field, out_dir, intent)
                    support_update = None
                if support_update:
                    field = field.model_copy(update={
                        "support": {**dict(field.support), **support_update},
                        "axes": {**dict(field.axes), **support_update},
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
                if op.upper() == "RN" or field.kind == "intensive_density":
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
