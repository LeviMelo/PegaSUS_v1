from __future__ import annotations

import re
from typing import Any

from pegasus.sidra.facts import normalize_flat_records_to_facts
from pegasus.sidra.schemas import SIDRAFactRow


def _is_header_row(row: dict[str, Any]) -> bool:
    if row.get("header_row") is True:
        return True

    value = str(row.get("V", row.get("value", ""))).strip().casefold()
    if value in {"valor", "value", "v"}:
        return True

    return str(row.get("D1C", "")).casefold() in {
        "município (código)",
        "municipio (codigo)",
        "localidade (código)",
        "localidade (codigo)",
    }


def _sidra_level_from_nc(value: Any, fallback: str | None = None) -> str:
    if value is None:
        return fallback or ""

    text = str(value).strip()
    if text.upper().startswith("N"):
        return text.upper()

    if text.isdigit():
        return f"N{text}"

    return fallback or text


def _ordered_classification_ids(chunk_request: dict[str, Any]) -> list[str]:
    classifications = chunk_request.get("classifications") or {}
    ids = [str(x) for x in classifications.keys()]
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def _header_dimension_names(rows: list[dict[str, Any]]) -> dict[int, str]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not _is_header_row(row):
            continue

        names: dict[int, str] = {}
        for key, value in row.items():
            m = re.fullmatch(r"D(\d+)N", str(key))
            if m and value is not None:
                names[int(m.group(1))] = str(value)
        return names

    return {}


def _classification_pairs_from_flat(
    row: dict[str, Any],
    *,
    chunk_request: dict[str, Any],
    header_names: dict[int, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if "classification_tuple" in row or "category_tuple" in row:
        raw_cls = row.get("classification_tuple") or []
        raw_cat = row.get("category_tuple") or []
        return (
            [tuple(map(str, x)) for x in raw_cls],
            [tuple(map(str, x)) for x in raw_cat],
        )

    class_ids = _ordered_classification_ids(chunk_request)
    classification_pairs: list[tuple[str, str]] = []
    category_pairs: list[tuple[str, str]] = []

    # Real SIDRA view=flat layout, as observed:
    # D1 = locality, D2 = period, D3 = variable, D4+ = classifications.
    for offset, cls_id in enumerate(class_ids):
        dim_idx = 4 + offset
        code = row.get(f"D{dim_idx}C")
        if code is None:
            continue

        classification_name = header_names.get(dim_idx, "")
        classification_pairs.append((cls_id, classification_name))
        category_pairs.append((cls_id, str(code)))

    return classification_pairs, category_pairs


def flat_response_to_records(payload: Any, *, table_id: str, chunk_request: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("resultados") or payload.get("values") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    rows = [row for row in rows if isinstance(row, dict)]
    header_names = _header_dimension_names(rows)

    records: list[dict[str, Any]] = []

    for row in rows:
        if _is_header_row(row):
            continue

        cls_pairs, cat_pairs = _classification_pairs_from_flat(
            row,
            chunk_request=chunk_request,
            header_names=header_names,
        )

        record = {
            "table_id": str(row.get("table_id") or row.get("agregado") or table_id),

            # Real SIDRA flat response:
            # MC/MN are measurement-unit code/name. Variable is D3C/D3N.
            "variable_id": str(row.get("variable_id") or row.get("variable") or row.get("D3C") or (chunk_request.get("variables") or [""])[0]),

            # Real SIDRA flat response:
            # D2C/D2N are period code/name.
            "period": str(row.get("period") or row.get("periodo") or row.get("D2C") or (chunk_request.get("periods") or [""])[0]),

            # Real SIDRA flat response:
            # NC is territorial-level code such as 6. Convert to N6.
            "locality_level": str(row.get("locality_level") or _sidra_level_from_nc(row.get("NC"), chunk_request.get("locality_level"))),

            # Real SIDRA flat response:
            # D1C is locality ID. NN is territorial-level name, not locality.
            "locality_id": str(row.get("locality_id") or row.get("localidade") or row.get("D1C") or ""),

            "classification_tuple": cls_pairs,
            "category_tuple": cat_pairs,
            "value": row.get("value", row.get("V")),
            "unit": row.get("unit") or row.get("MN"),
            "header_row": False,
        }

        # Preserve explicit fixture-style records.
        for key in [
            "variable_id",
            "period",
            "locality_level",
            "locality_id",
            "classification_tuple",
            "category_tuple",
            "value",
            "unit",
        ]:
            if key in row:
                record[key] = row[key]

        records.append(record)

    return records


def normalize_sidra_payload_to_facts(
    payload: Any,
    *,
    table_id: str,
    request_hash: str,
    metadata_hash: str,
    chunk_request: dict[str, Any],
    unit_by_variable: dict[str, str | None] | None = None,
    fetched_at: str | None = None,
) -> list[SIDRAFactRow]:
    records = flat_response_to_records(payload, table_id=table_id, chunk_request=chunk_request)
    return normalize_flat_records_to_facts(
        records,
        table_id=table_id,
        request_hash=request_hash,
        metadata_hash=metadata_hash,
        unit_by_variable=unit_by_variable,
        fetched_at=fetched_at,
    )
