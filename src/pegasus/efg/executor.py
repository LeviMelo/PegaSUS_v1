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
            df = pl.DataFrame({
                "field_id": [field.id],
                "field_name": [field.name],
                "operator": [field.operator or "scalar_support"],
                VALUE_COLUMN: [value],
            })
            return _write(output_dir / f"{field.id}.parquet", df)
    return None


def _count_tensor(field: FieldNode, source: Path, output_dir: Path) -> tuple[Path, int]:
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
    return _write(output_dir / f"{field.id}.parquet", out)


def _sum_tensor(field: FieldNode, source: Path, column: str, output_dir: Path) -> tuple[Path, int]:
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
    return _write(output_dir / f"{field.id}.parquet", out)


def _source_field_tensor(field: FieldNode, source: Path, column: str, output_dir: Path) -> tuple[Path, int]:
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
    return _write(output_dir / f"{field.id}.parquet", out)


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


def _compute_rn_ratio(field: FieldNode, parents_by_id: dict[str, FieldNode], output_dir: Path) -> tuple[Path, int]:
    parent_ids = list(field.lineage.parent_ids or [])
    if len(parent_ids) < 2:
        raise ValueError(f"RN field {field.id} requires numerator and denominator parents")
    numerator = parents_by_id[parent_ids[0]]
    denominator = parents_by_id[parent_ids[1]]
    n = _load_parent_tensor(numerator).rename({VALUE_COLUMN: "value_numerator"})
    d = _load_parent_tensor(denominator).rename({VALUE_COLUMN: "value_denominator"})
    keys = _join_keys(n, d)

    if not keys:
        raise RuntimeError(
            "RN operator requires at least one intersecting support axis; "
            "cross-join is forbidden to prevent OOM and indicates failed Δ support alignment."
        )
    joined = n.join(d, on=keys, how="inner", suffix="_denominator")

    out = joined.with_columns(
        pl.when(pl.col("value_denominator") > 0)
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
    return _write(output_dir / f"{field.id}.parquet", out)



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


def _compute_bridge_tensor(field: FieldNode, parents_by_id: dict[str, FieldNode], output_dir: Path) -> tuple[Path, int]:
    parent_ids = list(field.lineage.parent_ids or [])
    if not parent_ids:
        out = pl.DataFrame({
            "field_id": [field.id],
            "field_name": [field.name],
            "operator": [field.operator or "bridge"],
            VALUE_COLUMN: [None],
        }).cast({VALUE_COLUMN: pl.Float64})
        return _write(output_dir / f"{field.id}.parquet", out)

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
        return _write(output_dir / f"{field.id}.parquet", out)

    # Fallback for Bridge_R / unary bridges
    parent = parents_by_id[parent_ids[0]]
    df = _load_parent_tensor(parent)
    out = df.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "Bridge_R").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


def _execute_non_rn(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    scalar = _scalar_tensor(field, output_dir)
    if scalar is not None:
        return scalar

    source = _source_path(field)
    if source is None:
        raise ValueError("no source parquet path and no scalar support value")
    head = pl.read_parquet(source, n_rows=25)
    column = _source_column(field, head)
    op = str(field.operator or "").lower()

    if op in {"count_measure", "count", "event_count"} or field.unit in {"counts", "count"}:
        return _count_tensor(field, source, output_dir)
    if column is not None and field.aggregation in {"additive", "statistical_functional", "weighted_mean"}:
        return _sum_tensor(field, source, column, output_dir)
    if column is not None:
        return _source_field_tensor(field, source, column, output_dir)

    return _count_tensor(field, source, output_dir)


def execute_efg_result(
    efg: EFGResult,
    *,
    output_dir: str | Path,
    require_materialized: bool = True,
) -> tuple[EFGResult, EFGExecutionReport]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields_by_id: dict[str, FieldNode] = {field.id: field for field in efg.fields}
    executed: list[ExecutedField] = []

    # Fixed-point execution: source/scalar fields first, then RN and bridges.
    pending = set(fields_by_id)
    for _ in range(max(2, len(fields_by_id) + 1)):
        progressed = False
        for field_id in list(pending):
            field = fields_by_id[field_id]
            op = str(field.operator or "")
            try:
                if op.upper() == "RN" or field.kind == "intensive_density":
                    path, rows = _compute_rn_ratio(field, fields_by_id, out_dir)
                elif op.startswith("Bridge") or "bridge" in op.lower() or field.kind in {"bridge_module", "bridge_divergence"}:
                    path, rows = _compute_bridge_tensor(field, fields_by_id, out_dir)
                else:
                    path, rows = _execute_non_rn(field, out_dir)
                new_field = _materialized(field, path)
                fields_by_id[field_id] = new_field
                executed.append(ExecutedField(new_field, "success", str(path), rows))
                pending.remove(field_id)
                progressed = True
            except Exception as exc:
                # RN may be waiting for parent tensors. Keep it pending until the next pass.
                if op.upper() == "RN" or field.kind == "intensive_density":
                    continue
                executed.append(ExecutedField(field, "blocked", None, 0, f"{type(exc).__name__}: {exc}"))
                pending.remove(field_id)
                progressed = True
        if not pending or not progressed:
            break

    for field_id in sorted(pending):
        field = fields_by_id[field_id]
        executed.append(ExecutedField(field, "blocked", None, 0, "parents_not_materialized_or_operator_not_executable"))

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
