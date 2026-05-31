from __future__ import annotations

from pydantic import BaseModel, Field

from pegasus.datasus.profile import ColumnProfile, DataFrameProfile


class ColumnComparison(BaseModel):
    column: str
    raw_present: bool
    processed_present: bool
    raw_dtype: str | None = None
    processed_dtype: str | None = None
    raw_missing_rate: float | None = None
    processed_missing_rate: float | None = None
    raw_n_unique: int | None = None
    processed_n_unique: int | None = None
    raw_observed_kind: str | None = None
    processed_observed_kind: str | None = None
    flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SchemaComparison(BaseModel):
    source_system: str
    raw_n_rows: int
    processed_n_rows: int
    raw_only_columns: list[str]
    processed_only_columns: list[str]
    common_columns: list[str]
    column_comparisons: list[ColumnComparison]
    warnings: list[str] = Field(default_factory=list)


def _by_column(profile: DataFrameProfile) -> dict[str, ColumnProfile]:
    return {column.column: column for column in profile.columns}


def compare_profiles(
    raw: DataFrameProfile,
    processed: DataFrameProfile,
    *,
    missing_rate_delta_threshold: float = 0.05,
    unique_delta_ratio_threshold: float = 0.50,
) -> SchemaComparison:
    raw_cols = _by_column(raw)
    processed_cols = _by_column(processed)

    raw_names = set(raw_cols)
    processed_names = set(processed_cols)

    raw_only = sorted(raw_names - processed_names)
    processed_only = sorted(processed_names - raw_names)
    common = sorted(raw_names & processed_names)

    warnings: list[str] = []
    comparisons: list[ColumnComparison] = []

    if raw.n_rows != processed.n_rows:
        warnings.append(
            f"row_count_changed:raw={raw.n_rows};processed={processed.n_rows}"
        )

    for column in raw_only:
        comparisons.append(
            ColumnComparison(
                column=column,
                raw_present=True,
                processed_present=False,
                raw_dtype=raw_cols[column].dtype_observed,
                flags=["raw_only"],
                warnings=["raw_column_absent_from_processed"],
            )
        )

    for column in processed_only:
        comparisons.append(
            ColumnComparison(
                column=column,
                raw_present=False,
                processed_present=True,
                processed_dtype=processed_cols[column].dtype_observed,
                flags=["processed_only"],
                warnings=["processed_column_absent_from_raw"],
            )
        )

    for column in common:
        r = raw_cols[column]
        p = processed_cols[column]

        flags: list[str] = []
        col_warnings: list[str] = []

        if r.dtype_observed != p.dtype_observed:
            flags.append("dtype_changed")

        missing_delta = p.missing_rate - r.missing_rate
        if abs(missing_delta) >= missing_rate_delta_threshold:
            flags.append("missingness_changed")
            col_warnings.append(
                f"missingness_changed:raw={r.missing_rate:.4f};"
                f"processed={p.missing_rate:.4f}"
            )

        if r.n_unique > 0:
            unique_delta_ratio = abs(p.n_unique - r.n_unique) / r.n_unique
            if unique_delta_ratio >= unique_delta_ratio_threshold:
                flags.append("cardinality_changed")
                col_warnings.append(
                    f"cardinality_changed:raw={r.n_unique};processed={p.n_unique}"
                )

        if r.observed_kind != p.observed_kind:
            flags.append("observed_kind_changed")
            col_warnings.append(
                f"observed_kind_changed:raw={r.observed_kind};"
                f"processed={p.observed_kind}"
            )

        if p.n_nonblank < r.n_nonblank and p.n_unique > r.n_unique:
            flags.append("fewer_nonblank_but_more_unique")
            col_warnings.append("processed_not_simple_decode_inspect")

        comparisons.append(
            ColumnComparison(
                column=column,
                raw_present=True,
                processed_present=True,
                raw_dtype=r.dtype_observed,
                processed_dtype=p.dtype_observed,
                raw_missing_rate=r.missing_rate,
                processed_missing_rate=p.missing_rate,
                raw_n_unique=r.n_unique,
                processed_n_unique=p.n_unique,
                raw_observed_kind=r.observed_kind,
                processed_observed_kind=p.observed_kind,
                flags=flags,
                warnings=col_warnings,
            )
        )

    return SchemaComparison(
        source_system=raw.source_system,
        raw_n_rows=raw.n_rows,
        processed_n_rows=processed.n_rows,
        raw_only_columns=raw_only,
        processed_only_columns=processed_only,
        common_columns=common,
        column_comparisons=comparisons,
        warnings=warnings,
    )