from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.datasus.io import write_parquet


def sidra_population_facts_to_denominator_table(
    facts: pl.DataFrame,
    *,
    table_id: str | None = None,
    variable_id: str | None = None,
    source_label: str = "SIDRA_population",
) -> pl.DataFrame:
    required = {"period", "locality_id", "value", "value_state"}
    missing = sorted(required - set(facts.columns))

    if missing:
        raise ValueError(f"SIDRA facts table missing required columns: {missing}")

    df = facts

    if table_id is not None:
        df = df.filter(pl.col("table_id") == table_id)

    if variable_id is not None:
        df = df.filter(pl.col("variable_id") == variable_id)

    valid = df.filter(
        pl.col("period").is_not_null()
        & pl.col("locality_id").is_not_null()
        & pl.col("value").is_not_null()
        & pl.col("value_state").str.starts_with("valid")
    )

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
    )

    return out.sort(["year", "municipality_cod7"])


def write_sidra_population_denominator_table(
    *,
    facts_path: str | Path,
    output_path: str | Path,
    table_id: str | None = None,
    variable_id: str | None = None,
    source_label: str = "SIDRA_population",
) -> Path:
    facts = pl.read_parquet(facts_path)

    population = sidra_population_facts_to_denominator_table(
        facts,
        table_id=table_id,
        variable_id=variable_id,
        source_label=source_label,
    )

    write_parquet(population, output_path)
    return Path(output_path)