from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from pegasus.datasus.io import write_json, write_parquet


class SIDRAPopulationProjectionReport(BaseModel):
    input_facts_path: str | None = None
    output_path: str | None = None

    table_id_filter: str | None = None
    variable_id_filter: str | None = None
    required_locality_level_id: str | None = "6"

    n_input_facts: int
    n_after_table_filter: int
    n_after_variable_filter: int
    n_after_locality_level_filter: int
    n_after_value_state_filter: int
    n_output_rows: int

    excluded_by_table_filter: int
    excluded_by_variable_filter: int
    excluded_by_locality_level: int
    excluded_by_value_state: int

    locality_level_counts: dict[str, int] = Field(default_factory=dict)
    value_state_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    if df.is_empty() or column not in df.columns:
        return {}

    rows = (
        df.group_by(column)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .to_dicts()
    )

    return {str(row[column]): int(row["n"]) for row in rows}


def sidra_population_facts_to_denominator_table(
    facts: pl.DataFrame,
    *,
    table_id: str | None = None,
    variable_id: str | None = None,
    source_label: str = "SIDRA_population",
    required_locality_level_id: str | None = "6",
) -> tuple[pl.DataFrame, SIDRAPopulationProjectionReport]:
    required = {"period", "locality_id", "value", "value_state"}
    missing = sorted(required - set(facts.columns))

    if missing:
        raise ValueError(f"SIDRA facts table missing required columns: {missing}")

    warnings: list[str] = []

    n_input = facts.height
    df = facts

    if table_id is not None:
        df = df.filter(pl.col("table_id") == table_id)
    n_after_table = df.height

    if variable_id is not None:
        df = df.filter(pl.col("variable_id") == variable_id)
    n_after_variable = df.height

    if required_locality_level_id is not None:
        if "locality_level_id" not in df.columns:
            raise ValueError("SIDRA facts table lacks locality_level_id.")
        df = df.filter(pl.col("locality_level_id") == required_locality_level_id)
    n_after_level = df.height

    valid = df.filter(
        pl.col("period").is_not_null()
        & pl.col("locality_id").is_not_null()
        & pl.col("value").is_not_null()
        & pl.col("value_state").str.starts_with("valid")
    )
    n_after_value_state = valid.height

    if n_after_level == 0:
        warnings.append("no_rows_after_locality_level_filter")

    if n_after_value_state == 0:
        warnings.append("no_valid_population_values_after_value_state_filter")

    out = valid.select(
        [
            pl.col("period").cast(pl.Int32, strict=False).alias("year"),
            pl.col("locality_id").cast(pl.Utf8).alias("municipality_cod7"),
            pl.col("locality_id").cast(pl.Utf8).str.slice(0, 6).alias("municipality_cod6"),
            pl.col("value").cast(pl.Float64).alias("population"),
            pl.lit("valid").alias("population_state"),
            pl.lit(source_label).alias("denominator_source"),
            pl.lit("official").alias("provenance"),
            (
                pl.col("table_id").cast(pl.Utf8)
                + pl.lit(":")
                + pl.col("variable_id").cast(pl.Utf8)
            ).alias("sidra_measure_id"),
            pl.col("locality_name").cast(pl.Utf8).alias("municipality_name"),
        ]
    ).sort(["year", "municipality_cod7"])

    duplicate_keys = (
        out.group_by(["year", "municipality_cod7"])
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )

    if duplicate_keys.height:
        warnings.append(f"duplicate_population_keys:{duplicate_keys.height}")

    report = SIDRAPopulationProjectionReport(
        table_id_filter=table_id,
        variable_id_filter=variable_id,
        required_locality_level_id=required_locality_level_id,
        n_input_facts=n_input,
        n_after_table_filter=n_after_table,
        n_after_variable_filter=n_after_variable,
        n_after_locality_level_filter=n_after_level,
        n_after_value_state_filter=n_after_value_state,
        n_output_rows=out.height,
        excluded_by_table_filter=n_input - n_after_table,
        excluded_by_variable_filter=n_after_table - n_after_variable,
        excluded_by_locality_level=n_after_variable - n_after_level,
        excluded_by_value_state=n_after_level - n_after_value_state,
        locality_level_counts=_counts(facts, "locality_level_id"),
        value_state_counts=_counts(facts, "value_state"),
        warnings=warnings,
    )

    return out, report


def write_sidra_population_denominator_table(
    *,
    facts_path: str | Path,
    output_path: str | Path,
    table_id: str | None = None,
    variable_id: str | None = None,
    source_label: str = "SIDRA_population",
    required_locality_level_id: str | None = "6",
    report_path: str | Path | None = None,
) -> tuple[Path, SIDRAPopulationProjectionReport]:
    facts_path = Path(facts_path)
    output_path = Path(output_path)

    facts = pl.read_parquet(facts_path)

    population, report = sidra_population_facts_to_denominator_table(
        facts,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
        required_locality_level_id=required_locality_level_id,
    )

    write_parquet(population, output_path)

    final_report = report.model_copy(
        update={
            "input_facts_path": str(facts_path),
            "output_path": str(output_path),
        }
    )

    if report_path is None:
        report_path = output_path.with_suffix(".projection_report.json")

    write_json(report_path, final_report)

    return output_path, final_report