from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel


def read_table(path: str | Path, *, csv_all_as_string: bool = True) -> pl.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pl.read_parquet(path)

    if suffix == ".csv":
        return pl.read_csv(
            path,
            infer_schema=False if csv_all_as_string else True,
            infer_schema_length=10_000,
            ignore_errors=False,
        )

    if suffix in {".tsv", ".txt"}:
        return pl.read_csv(
            path,
            separator="\t",
            infer_schema=False if csv_all_as_string else True,
            infer_schema_length=10_000,
            ignore_errors=False,
        )

    raise ValueError(f"Unsupported table format: {path}")


def write_parquet(df: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def write_json(path: str | Path, payload: BaseModel | dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    else:
        data = payload

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)