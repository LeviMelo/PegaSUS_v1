from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.sidra.schemas import SIDRAFactRow


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_sidra_value(raw: Any) -> tuple[float | None, str]:
    if raw is None:
        return None, "blank"

    text = str(raw).strip()
    if text == "":
        return None, "blank"

    if text in {"-", "—", "–"}:
        return None, "dash_zero_or_nil"

    lowered = text.casefold()
    if lowered in {"x", "...", "na", "n/a"}:
        return None, "not_available"

    if lowered in {"..", "…"}:
        return None, "suppressed_or_unidentified"

    normalized = text.replace(".", "").replace(",", ".") if "," in text else text

    try:
        return float(normalized), "numeric"
    except ValueError:
        return None, "non_numeric_symbol"


def _tuple_from_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return tuple(tuple(map(str, x)) for x in value)
    if isinstance(value, list):
        return tuple(tuple(map(str, x)) for x in value)
    if isinstance(value, dict):
        return tuple((str(k), str(v)) for k, v in sorted(value.items()))
    return tuple()


def normalize_flat_records_to_facts(
    records: list[dict[str, Any]],
    *,
    table_id: str,
    request_hash: str,
    metadata_hash: str,
    unit_by_variable: dict[str, str | None] | None = None,
    fetched_at: str | None = None,
) -> list[SIDRAFactRow]:
    unit_by_variable = unit_by_variable or {}
    fetched_at = fetched_at or utc_now()
    facts: list[SIDRAFactRow] = []

    for record in records:
        if record.get("header_row") is True:
            status = "header_row"
            value_numeric = None
        else:
            value_numeric, status = classify_sidra_value(record.get("value"))

        variable_id = str(record.get("variable_id") or record.get("variable") or "")
        if not variable_id:
            continue

        unit = unit_by_variable.get(variable_id)
        if unit is None and record.get("unit") is not None:
            unit = str(record.get("unit"))

        facts.append(
            SIDRAFactRow(
                table_id=str(record.get("table_id") or table_id),
                variable_id=variable_id,
                period=str(record.get("period") or ""),
                locality_level=str(record.get("locality_level") or ""),
                locality_id=str(record.get("locality_id") or ""),
                classification_tuple=_tuple_from_pairs(record.get("classification_tuple")),
                category_tuple=_tuple_from_pairs(record.get("category_tuple")),
                value_raw=None if record.get("value") is None else str(record.get("value")),
                value_numeric=value_numeric,
                value_status=status,
                unit=unit,
                request_hash=request_hash,
                metadata_hash=metadata_hash,
                fetched_at=fetched_at,
            )
        )

    return facts


def facts_to_frame(facts: list[SIDRAFactRow]) -> pl.DataFrame:
    rows = []
    for fact in facts:
        row = fact.model_dump(mode="json")
        row["classification_tuple"] = json.dumps(row["classification_tuple"], ensure_ascii=False)
        row["category_tuple"] = json.dumps(row["category_tuple"], ensure_ascii=False)
        rows.append(row)

    if not rows:
        return pl.DataFrame(
            schema={
                "table_id": pl.Utf8,
                "variable_id": pl.Utf8,
                "period": pl.Utf8,
                "locality_level": pl.Utf8,
                "locality_id": pl.Utf8,
                "classification_tuple": pl.Utf8,
                "category_tuple": pl.Utf8,
                "value_raw": pl.Utf8,
                "value_numeric": pl.Float64,
                "value_status": pl.Utf8,
                "unit": pl.Utf8,
                "request_hash": pl.Utf8,
                "metadata_hash": pl.Utf8,
                "fetched_at": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def write_facts_parquet(
    facts: list[SIDRAFactRow],
    *,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    facts_to_frame(facts).write_parquet(output_path, compression="zstd")
    return output_path


def normalize_fixture_json_to_facts(
    *,
    input_path: str | Path,
    output_path: str | Path,
    table_id: str,
    unit_by_variable: dict[str, str | None] | None = None,
) -> Path:
    """Normalize a checked-in flat SIDRA fixture JSON file to facts parquet."""
    input_path = Path(input_path)
    payload = input_path.read_bytes()
    records = json.loads(payload.decode("utf-8"))
    if not isinstance(records, list):
        raise ValueError("SIDRA fixture JSON must contain a list of flat records.")
    fixture_hash = hashlib.sha256(payload).hexdigest()
    facts = normalize_flat_records_to_facts(
        records,
        table_id=table_id,
        request_hash=f"fixture_request:{fixture_hash}",
        metadata_hash=f"fixture_metadata:{fixture_hash}",
        unit_by_variable=unit_by_variable,
        fetched_at="fixture",
    )
    return write_facts_parquet(facts, output_path=output_path)
