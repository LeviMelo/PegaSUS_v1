from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def _scan(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.scan_csv(path, infer_schema_length=1000, ignore_errors=True)
    if suffix in {".json", ".ndjson"}:
        return pl.scan_ndjson(path)
    raise ValueError(f"Unsupported profile format: {path}")


def _top_values(df: pl.DataFrame, column: str, limit: int = 20) -> list[dict[str, Any]]:
    vc = (
        df.select(pl.col(column).cast(pl.Utf8).alias(column))
        .group_by(column)
        .len()
        .sort("len", descending=True)
        .head(limit)
    )
    return vc.to_dicts()


def profile_table(path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    lf = _scan(path)
    df = lf.collect()

    profile: dict[str, Any] = {
        "path": str(path),
        "row_count": df.height,
        "column_count": len(df.columns),
        "columns": [],
    }

    for col in df.columns:
        series = df[col]
        non_null = series.drop_nulls()
        col_profile: dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "row_count": df.height,
            "missing_count": int(series.null_count()),
            "missing_rate": float(series.null_count() / df.height) if df.height else None,
            "nonblank_count": int(non_null.len()),
            "unique_count": int(series.n_unique()),
            "unique_rate": float(series.n_unique() / df.height) if df.height else None,
            "top_values": _top_values(df, col),
        }

        as_utf8 = series.cast(pl.Utf8, strict=False)
        lengths = as_utf8.str.len_chars()
        col_profile["min_length"] = int(lengths.min()) if lengths.len() and lengths.min() is not None else None
        col_profile["max_length"] = int(lengths.max()) if lengths.len() and lengths.max() is not None else None

        numeric = as_utf8.str.replace(",", ".").cast(pl.Float64, strict=False)
        numeric_valid = numeric.drop_nulls()
        col_profile["numeric_parse_count"] = int(numeric_valid.len())
        col_profile["numeric_parse_rate"] = float(numeric_valid.len() / df.height) if df.height else None
        col_profile["numeric_min"] = float(numeric_valid.min()) if numeric_valid.len() else None
        col_profile["numeric_max"] = float(numeric_valid.max()) if numeric_valid.len() else None

        profile["columns"].append(col_profile)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    return profile
