from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field


class SIDRAFactsDiagnostics(BaseModel):
    facts_path: str
    n_rows: int
    n_unique_tables: int
    n_unique_variables: int
    n_unique_periods: int
    n_unique_localities: int

    table_counts: dict[str, int] = Field(default_factory=dict)
    variable_counts: dict[str, int] = Field(default_factory=dict)
    period_counts: dict[str, int] = Field(default_factory=dict)
    locality_level_counts: dict[str, int] = Field(default_factory=dict)
    value_state_counts: dict[str, int] = Field(default_factory=dict)

    n_null_value: int
    n_valid_value: int
    n_duplicate_measure_keys: int
    warnings: list[str] = Field(default_factory=list)


def _counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns or df.is_empty():
        return {}

    rows = (
        df.group_by(column)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .to_dicts()
    )

    return {str(row[column]): int(row["n"]) for row in rows}


def diagnose_sidra_facts(facts_path: str | Path) -> SIDRAFactsDiagnostics:
    facts_path = Path(facts_path)
    df = pl.read_parquet(facts_path)

    warnings: list[str] = []

    required = {
        "table_id",
        "variable_id",
        "period",
        "locality_id",
        "locality_level_id",
        "value",
        "value_state",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        warnings.append("missing_expected_columns:" + ",".join(missing))

    duplicate_keys = pl.DataFrame()

    if not missing:
        duplicate_keys = (
            df.group_by(["table_id", "variable_id", "period", "locality_id"])
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") > 1)
        )

        if duplicate_keys.height:
            warnings.append(f"duplicate_sidra_measure_keys:{duplicate_keys.height}")

    n_valid = (
        df.filter(pl.col("value_state").str.starts_with("valid")).height
        if "value_state" in df.columns
        else 0
    )

    n_null_value = (
        df.filter(pl.col("value").is_null()).height
        if "value" in df.columns
        else 0
    )

    return SIDRAFactsDiagnostics(
        facts_path=str(facts_path),
        n_rows=df.height,
        n_unique_tables=df["table_id"].n_unique() if "table_id" in df.columns else 0,
        n_unique_variables=df["variable_id"].n_unique() if "variable_id" in df.columns else 0,
        n_unique_periods=df["period"].n_unique() if "period" in df.columns else 0,
        n_unique_localities=df["locality_id"].n_unique() if "locality_id" in df.columns else 0,
        table_counts=_counts(df, "table_id"),
        variable_counts=_counts(df, "variable_id"),
        period_counts=_counts(df, "period"),
        locality_level_counts=_counts(df, "locality_level_id"),
        value_state_counts=_counts(df, "value_state"),
        n_null_value=n_null_value,
        n_valid_value=n_valid,
        n_duplicate_measure_keys=duplicate_keys.height,
        warnings=warnings,
    )