"""SHE zero-variance and all-missing exclusion gate.

This module is intentionally source-agnostic. It inspects already-materialized
local source artifacts and classifies columns before they can become substrate
candidates. The gate is conservative: 100% missing columns, constant columns,
identifier/hash/raw payload columns, and structurally unsupported columns are
excluded from analytical substrate candidates and remain auditable only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

NULL_TOKENS: frozenset[str] = frozenset({"", "na", "nan", "null", "none", "missing"})
IDENTIFIER_SUFFIXES: tuple[str, ...] = (
    "_id",
    "_hash",
    "_json",
    "_raw",
    "raw_json",
    "row_hash",
    "raw_record_hash",
    "processed_record_hash",
    "source_manifest_hash",
)
IDENTIFIER_NAMES: frozenset[str] = frozenset({
    "event_id",
    "admission_id",
    "facility_id",
    "raw_json",
    "row_hash",
    "raw_record_hash",
    "processed_record_hash",
    "source_manifest_hash",
})
STATE_MARKERS: tuple[str, ...] = ("_state", "_parse_state", "_states_json", "_state_json")


class ZeroVarianceError(ValueError):
    """Raised when a substrate variance gate cannot inspect an artifact."""


@dataclass(frozen=True)
class ColumnVarianceProfile:
    column: str
    dtype: str
    row_count: int
    non_null_count: int
    missing_count: int
    missing_rate: float | None
    unique_non_null_count: int
    constant_value_repr: str | None
    numeric_parse_count: int
    numeric_min: float | None
    numeric_max: float | None
    structural_role: str
    admissible: bool
    exclusion_reason: str | None
    warnings: tuple[str, ...]

    @property
    def all_missing(self) -> bool:
        return self.row_count > 0 and self.non_null_count == 0

    @property
    def constant(self) -> bool:
        return self.non_null_count > 0 and self.unique_non_null_count <= 1

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["all_missing"] = self.all_missing
        payload["constant"] = self.constant
        return payload


@dataclass(frozen=True)
class TableVarianceProfile:
    path: str
    row_count: int
    column_count: int
    admissible_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    all_missing_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    structural_only_columns: tuple[str, ...]
    profiles: tuple[ColumnVarianceProfile, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "admissible_columns": list(self.admissible_columns),
            "excluded_columns": list(self.excluded_columns),
            "all_missing_columns": list(self.all_missing_columns),
            "constant_columns": list(self.constant_columns),
            "structural_only_columns": list(self.structural_only_columns),
            "profiles": [p.as_manifest() for p in self.profiles],
        }


def _read_table(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise ZeroVarianceError(f"Source artifact not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(p)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(p, infer_schema_length=1000, ignore_errors=False)
    if suffix in {".json", ".ndjson"}:
        return pl.read_ndjson(p)
    raise ZeroVarianceError(f"Unsupported substrate artifact format: {p}")


def _blank_normalized_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()


def _non_missing_expr(column: str) -> pl.Expr:
    return pl.col(column).is_not_null() & (~_blank_normalized_expr(column).is_in(list(NULL_TOKENS)))


def classify_structural_role(column: str) -> str:
    name = column.strip()
    lower = name.lower()
    if lower in IDENTIFIER_NAMES or lower.endswith(IDENTIFIER_SUFFIXES):
        return "identifier_or_raw_payload"
    if any(lower.endswith(marker) for marker in STATE_MARKERS):
        return "state_or_quality_marker"
    if lower.startswith("raw_") or lower.endswith("_raw_json"):
        return "raw_payload"
    if lower.endswith("_code") or lower.endswith("_norm") or lower.endswith("_icd_norm"):
        return "categorical_or_code"
    if lower.endswith("_flag") or lower.startswith("is_"):
        return "boolean_measure"
    if any(token in lower for token in ("count", "total", "days", "cost", "weight", "age", "year", "month", "score", "value")):
        return "quantitative_measure"
    return "candidate_measure"


def _constant_repr(values: list[Any]) -> str | None:
    if not values:
        return None
    first = values[0]
    try:
        return json.dumps(first, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(first)


def _numeric_stats(series: pl.Series, non_missing: pl.Series) -> tuple[int, float | None, float | None]:
    if series.len() == 0:
        return 0, None, None
    try:
        numeric = series.cast(pl.Utf8, strict=False).str.replace_all(",", ".").cast(pl.Float64, strict=False)
        numeric = numeric.filter(non_missing)
        numeric_valid = numeric.drop_nulls()
        count = int(numeric_valid.len())
        if count == 0:
            return 0, None, None
        min_v = numeric_valid.min()
        max_v = numeric_valid.max()
        return count, float(min_v) if min_v is not None and math.isfinite(float(min_v)) else None, float(max_v) if max_v is not None and math.isfinite(float(max_v)) else None
    except Exception:
        return 0, None, None


def profile_column(df: pl.DataFrame, column: str) -> ColumnVarianceProfile:
    if column not in df.columns:
        raise ZeroVarianceError(f"Column not found for variance profiling: {column}")
    row_count = int(df.height)
    series = df[column]
    dtype = str(series.dtype)
    if row_count == 0:
        return ColumnVarianceProfile(
            column=column,
            dtype=dtype,
            row_count=0,
            non_null_count=0,
            missing_count=0,
            missing_rate=None,
            unique_non_null_count=0,
            constant_value_repr=None,
            numeric_parse_count=0,
            numeric_min=None,
            numeric_max=None,
            structural_role=classify_structural_role(column),
            admissible=False,
            exclusion_reason="empty_table",
            warnings=("empty_source_artifact",),
        )

    non_missing_mask = df.select(_non_missing_expr(column).alias("non_missing"))["non_missing"]
    non_null_count = int(non_missing_mask.sum())
    missing_count = row_count - non_null_count
    missing_rate = missing_count / float(row_count) if row_count else None
    non_missing = series.filter(non_missing_mask)
    # Native columnar unique count (C-speed) rather than json.dumps-ing every value
    # into a Python set -- the latter was ~100M json.dumps calls on the large SIH/
    # CNES tables and dominated the whole compile's she_build stage. Nested dtypes
    # (Struct/List) aren't hashable by n_unique in all polars versions, so fall back
    # to a string cast for those (equivalent for the <=1 constant test that matters).
    try:
        unique_non_null_count = int(non_missing.n_unique())
    except Exception:
        unique_non_null_count = int(non_missing.cast(pl.Utf8, strict=False).n_unique())
    structural_role = classify_structural_role(column)
    numeric_parse_count, numeric_min, numeric_max = _numeric_stats(series, non_missing_mask)

    warnings: list[str] = []
    exclusion_reason: str | None = None
    admissible = True

    if non_null_count == 0:
        admissible = False
        exclusion_reason = "all_missing"
        warnings.append("substrate_all_missing_excluded")
    elif unique_non_null_count <= 1:
        admissible = False
        exclusion_reason = "zero_variance_constant"
        warnings.append("substrate_zero_variance_excluded")
    elif structural_role in {"identifier_or_raw_payload", "raw_payload", "state_or_quality_marker"}:
        admissible = False
        exclusion_reason = "structural_or_audit_only"
        warnings.append("substrate_structural_column_excluded")

    return ColumnVarianceProfile(
        column=column,
        dtype=dtype,
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_non_null_count=unique_non_null_count,
        constant_value_repr=_constant_repr(non_missing.head(1).to_list()) if unique_non_null_count <= 1 else None,
        numeric_parse_count=numeric_parse_count,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        structural_role=structural_role,
        admissible=admissible,
        exclusion_reason=exclusion_reason,
        warnings=tuple(warnings),
    )


def profile_table_variance(path: str | Path, *, columns: Iterable[str] | None = None) -> TableVarianceProfile:
    p = Path(path)
    df = _read_table(p)
    selected = list(columns) if columns is not None else list(df.columns)
    profiles = tuple(profile_column(df, column) for column in selected if column in df.columns)
    admissible = tuple(p.column for p in profiles if p.admissible)
    excluded = tuple(p.column for p in profiles if not p.admissible)
    all_missing = tuple(p.column for p in profiles if p.exclusion_reason == "all_missing")
    constant = tuple(p.column for p in profiles if p.exclusion_reason == "zero_variance_constant")
    structural = tuple(p.column for p in profiles if p.exclusion_reason == "structural_or_audit_only")
    return TableVarianceProfile(
        path=str(p),
        row_count=int(df.height),
        column_count=len(df.columns),
        admissible_columns=admissible,
        excluded_columns=excluded,
        all_missing_columns=all_missing,
        constant_columns=constant,
        structural_only_columns=structural,
        profiles=profiles,
    )


def assert_no_zero_variance_admissible(profile: TableVarianceProfile) -> None:
    bad = [p.column for p in profile.profiles if p.admissible and (p.all_missing or p.constant)]
    if bad:
        raise AssertionError(f"Zero-variance/all-missing columns were admitted: {bad}")
