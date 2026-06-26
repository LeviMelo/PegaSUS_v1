from __future__ import annotations
from pegasus.storage import read_table

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import content_hash


TOTAL_CATEGORY_SET_9606 = {
    ("2", "6794"),       # Sexo total
    ("86", "95251"),     # Cor ou raça total
    ("287", "100362"),   # Idade total
}


@dataclass(frozen=True)
class SidraPopulationAnchor:
    field_id: str
    table_id: str
    variable_id: str
    period: str
    locality_level: str
    locality_id: str
    value: float
    unit: str
    request_hash: str
    metadata_hash: str
    classification_tuple: list[tuple[str, str]]
    category_tuple: list[tuple[str, str]]
    total_category_policy: str = "total_only"
    projection_operator: str = "Pi_Clsf_to_Axis/pi_bound_*"


def _loads_tuple(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return [tuple(map(str, x)) for x in parsed]


def _is_total_9606(row: dict[str, Any]) -> bool:
    try:
        cats = set(_loads_tuple(row["category_tuple"]))
    except Exception:
        return False
    return TOTAL_CATEGORY_SET_9606 <= cats



def load_sidra_population_totals_frame(facts_path: str | Path) -> pl.DataFrame:
    """Per-municipality resident-population totals from SIDRA 9606 (var 93).

    Returns one row per (municipality_cod6, year) carrying the Total
    sex/race/age population — the denominator panel a state or national run
    needs. The single-locality ``load_sidra_population_total_anchor`` only ever
    served single-city smoke; this is its multi-locality generalization."""
    df = pl.from_arrow(read_table(Path(facts_path)))
    total_ids = sorted({cat for _clsf, cat in TOTAL_CATEGORY_SET_9606})  # 100362, 6794, 95251
    cat = pl.col("category_tuple").cast(pl.Utf8)
    is_total = pl.all_horizontal([cat.str.contains(tid, literal=True) for tid in total_ids])
    filtered = df.filter(
        (pl.col("table_id").cast(pl.Utf8) == "9606")
        & (pl.col("variable_id").cast(pl.Utf8) == "93")
        & (pl.col("value_status").cast(pl.Utf8) == "numeric")
        & pl.col("value_numeric").is_not_null()
        & is_total
    )
    out = filtered.with_columns(
        pl.col("locality_id").cast(pl.Utf8).str.slice(0, 6).alias("municipality_cod6"),
        pl.col("period").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64, strict=False).alias("year"),
        pl.col("value_numeric").cast(pl.Float64).alias("value"),
    ).select(["municipality_cod6", "year", "value"]).unique(subset=["municipality_cod6", "year"])
    return out


def load_sidra_population_total_anchor(facts_path: str | Path) -> SidraPopulationAnchor:
    facts_path = Path(facts_path)
    df = pl.from_arrow(read_table(facts_path))

    required = {
        "table_id",
        "variable_id",
        "period",
        "locality_level",
        "locality_id",
        "classification_tuple",
        "category_tuple",
        "value_numeric",
        "value_status",
        "unit",
        "request_hash",
        "metadata_hash",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SIDRA facts file is missing required columns: {sorted(missing)}")

    candidates = []
    for row in df.to_dicts():
        if str(row["table_id"]) != "9606":
            continue
        if str(row["variable_id"]) != "93":
            continue
        if str(row["value_status"]) != "numeric":
            continue
        if row["value_numeric"] is None:
            continue
        if str(row.get("metadata_hash") or "") in {"", "metadata_unset"}:
            continue
        if not _is_total_9606(row):
            continue
        candidates.append(row)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one total SIDRA 9606 population anchor fact; found {len(candidates)}."
        )

    row = candidates[0]
    category_tuple = _loads_tuple(row["category_tuple"])
    classification_tuple = _loads_tuple(row["classification_tuple"])

    field_payload = {
        "kind": "SIDRAPopulationTotalAnchor",
        "facts_path": str(facts_path),
        "table_id": str(row["table_id"]),
        "variable_id": str(row["variable_id"]),
        "period": str(row["period"]),
        "locality_level": str(row["locality_level"]),
        "locality_id": str(row["locality_id"]),
        "category_tuple": category_tuple,
        "request_hash": str(row["request_hash"]),
        "metadata_hash": str(row["metadata_hash"]),
    }

    return SidraPopulationAnchor(
        field_id=content_hash(field_payload),
        table_id=str(row["table_id"]),
        variable_id=str(row["variable_id"]),
        period=str(row["period"]),
        locality_level=str(row["locality_level"]),
        locality_id=str(row["locality_id"]),
        value=float(row["value_numeric"]),
        unit=str(row["unit"] or "Pessoas"),
        request_hash=str(row["request_hash"]),
        metadata_hash=str(row["metadata_hash"]),
        classification_tuple=classification_tuple,
        category_tuple=category_tuple,
    )
