from __future__ import annotations

import json
import re
from typing import Any

import polars as pl


_DIM_CODE_RE = re.compile(r"^D(\d+)C$")
_DIM_NAME_RE = re.compile(r"^D(\d+)N$")


def parse_sidra_value(value: Any) -> tuple[float | None, str]:
    if value is None:
        return None, "missing"

    text = str(value).strip()

    if text == "":
        return None, "blank"

    if text == "-":
        return 0.0, "valid_zero_absolute"

    if text == "0":
        return 0.0, "valid_zero_rounded_or_exact"

    if text == "...":
        return None, "not_available"

    if text == "..":
        return None, "not_applicable"

    if text.upper() == "X":
        return None, "inhibited"

    normalized = text.replace(".", "").replace(",", ".")

    try:
        return float(normalized), "valid"
    except ValueError:
        return None, "unparseable"


def normalize_sidra_response(
    payload: Any,
    *,
    table_id: str,
    expected_variables: set[str] | None = None,
    expected_periods: set[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    if not isinstance(payload, list):
        raise ValueError("SIDRA payload must be a list.")

    if not payload:
        return _empty_facts(), ["sidra_empty_top_level_payload"]

    if _looks_like_flat_response(payload):
        return _normalize_flat_response(
            payload,
            table_id=table_id,
            expected_variables=expected_variables or set(),
            expected_periods=expected_periods or set(),
        )

    if _looks_like_nested_response(payload):
        return _normalize_nested_response(payload, table_id=table_id)

    return _empty_facts(), ["sidra_unrecognized_response_shape"]


def _empty_facts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "table_id": pl.Series([], dtype=pl.Utf8),
            "variable_id": pl.Series([], dtype=pl.Utf8),
            "variable_name": pl.Series([], dtype=pl.Utf8),
            "unit": pl.Series([], dtype=pl.Utf8),
            "period": pl.Series([], dtype=pl.Utf8),
            "locality_id": pl.Series([], dtype=pl.Utf8),
            "locality_name": pl.Series([], dtype=pl.Utf8),
            "locality_level_id": pl.Series([], dtype=pl.Utf8),
            "locality_level_name": pl.Series([], dtype=pl.Utf8),
            "classification_tuple_json": pl.Series([], dtype=pl.Utf8),
            "dimension_tuple_json": pl.Series([], dtype=pl.Utf8),
            "value_raw": pl.Series([], dtype=pl.Utf8),
            "value": pl.Series([], dtype=pl.Float64),
            "value_state": pl.Series([], dtype=pl.Utf8),
        }
    )


def _looks_like_flat_response(payload: list[Any]) -> bool:
    return all(isinstance(row, dict) for row in payload) and any(
        "V" in row for row in payload if isinstance(row, dict)
    )


def _looks_like_nested_response(payload: list[Any]) -> bool:
    return all(isinstance(row, dict) for row in payload) and any(
        "resultados" in row for row in payload if isinstance(row, dict)
    )


def _is_flat_header_row(row: dict[str, Any]) -> bool:
    value = str(row.get("V", "")).strip().lower()

    if value in {"valor", "v"}:
        return True

    # Header rows usually contain labels such as "Município (Código)" rather
    # than actual codes in DnC columns.
    d_code_values = [
        str(v).lower()
        for k, v in row.items()
        if _DIM_CODE_RE.match(k) and v is not None
    ]

    return any("código" in v or "codigo" in v for v in d_code_values)


def _dimension_pairs(row: dict[str, Any]) -> list[dict[str, str]]:
    dims: list[dict[str, str]] = []

    for key, code_value in row.items():
        match = _DIM_CODE_RE.match(key)
        if not match:
            continue

        idx = match.group(1)
        name_key = f"D{idx}N"

        dims.append(
            {
                "index": idx,
                "code": "" if code_value is None else str(code_value),
                "name": "" if row.get(name_key) is None else str(row.get(name_key)),
            }
        )

    dims.sort(key=lambda item: int(item["index"]))
    return dims


def _infer_variable_period_locality(
    dims: list[dict[str, str]],
    *,
    expected_variables: set[str],
    expected_periods: set[str],
) -> tuple[str, str, str, str, list[dict[str, str]]]:
    variable_id = ""
    variable_name = ""
    period = ""
    locality_id = ""
    locality_name = ""

    remaining: list[dict[str, str]] = []

    for dim in dims:
        code = dim["code"]
        name = dim["name"]

        if code in expected_variables and not variable_id:
            variable_id = code
            variable_name = name
            continue

        if code in expected_periods and not period:
            period = code
            continue

        if not period and re.fullmatch(r"\d{4}", code):
            period = code
            continue

        remaining.append(dim)

    if not locality_id and remaining:
        # In flat SIDRA output, the first non-variable/non-period dimension is
        # normally the territorial unit selected through localidades.
        locality = remaining.pop(0)
        locality_id = locality["code"]
        locality_name = locality["name"]

    if not variable_id:
        for dim in remaining:
            if "população" in dim["name"].lower() or "variável" in dim["name"].lower():
                variable_id = dim["code"]
                variable_name = dim["name"]
                remaining.remove(dim)
                break

    return variable_id, variable_name, period, locality_id, locality_name, remaining


def _normalize_flat_response(
    payload: list[Any],
    *,
    table_id: str,
    expected_variables: set[str],
    expected_periods: set[str],
) -> tuple[pl.DataFrame, list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for item in payload:
        if not isinstance(item, dict):
            warnings.append("sidra_flat_non_dict_row")
            continue

        if _is_flat_header_row(item):
            continue

        raw_value = item.get("V")
        parsed_value, value_state = parse_sidra_value(raw_value)

        dims = _dimension_pairs(item)

        variable_id, variable_name, period, locality_id, locality_name, rest_dims = (
            _infer_variable_period_locality(
                dims,
                expected_variables=expected_variables,
                expected_periods=expected_periods,
            )
        )

        locality_level_id = "" if item.get("NC") is None else str(item.get("NC"))
        locality_level_name = "" if item.get("NN") is None else str(item.get("NN"))
        unit = "" if item.get("MN") is None else str(item.get("MN"))

        rows.append(
            {
                "table_id": str(table_id),
                "variable_id": variable_id,
                "variable_name": variable_name,
                "unit": unit,
                "period": period,
                "locality_id": locality_id,
                "locality_name": locality_name,
                "locality_level_id": locality_level_id,
                "locality_level_name": locality_level_name,
                "classification_tuple_json": json.dumps(
                    rest_dims,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "dimension_tuple_json": json.dumps(
                    dims,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "value_raw": None if raw_value is None else str(raw_value),
                "value": parsed_value,
                "value_state": value_state,
            }
        )

    if not rows:
        return _empty_facts(), [*warnings, "sidra_flat_no_data_rows_after_header_removal"]

    return pl.DataFrame(rows), warnings


def _normalize_nested_response(
    payload: list[Any],
    *,
    table_id: str,
) -> tuple[pl.DataFrame, list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for variable_block in payload:
        if not isinstance(variable_block, dict):
            warnings.append("unexpected_variable_block_type")
            continue

        variable_id = str(variable_block.get("id", ""))
        variable_name = str(variable_block.get("variavel", ""))
        unit = str(variable_block.get("unidade", ""))

        resultados = variable_block.get("resultados", [])

        if not isinstance(resultados, list):
            warnings.append(f"unexpected_resultados_type:{variable_id}")
            continue

        for result in resultados:
            if not isinstance(result, dict):
                warnings.append(f"unexpected_result_type:{variable_id}")
                continue

            classifications = result.get("classificacoes", [])
            classification_tuple_json = json.dumps(
                classifications,
                ensure_ascii=False,
                sort_keys=True,
            )

            series = result.get("series", [])

            if not isinstance(series, list):
                warnings.append(f"unexpected_series_type:{variable_id}")
                continue

            for series_item in series:
                if not isinstance(series_item, dict):
                    warnings.append(f"unexpected_series_item_type:{variable_id}")
                    continue

                locality = series_item.get("localidade", {})
                serie = series_item.get("serie", {})

                if not isinstance(locality, dict) or not isinstance(serie, dict):
                    warnings.append(f"unexpected_locality_or_serie_type:{variable_id}")
                    continue

                locality_level = locality.get("nivel", {})
                if not isinstance(locality_level, dict):
                    locality_level = {}

                locality_id = str(locality.get("id", ""))
                locality_name = str(locality.get("nome", ""))
                locality_level_id = str(locality_level.get("id", ""))
                locality_level_name = str(locality_level.get("nome", ""))

                for period, raw_value in serie.items():
                    parsed_value, value_state = parse_sidra_value(raw_value)

                    rows.append(
                        {
                            "table_id": str(table_id),
                            "variable_id": variable_id,
                            "variable_name": variable_name,
                            "unit": unit,
                            "period": str(period),
                            "locality_id": locality_id,
                            "locality_name": locality_name,
                            "locality_level_id": locality_level_id,
                            "locality_level_name": locality_level_name,
                            "classification_tuple_json": classification_tuple_json,
                            "dimension_tuple_json": "[]",
                            "value_raw": None if raw_value is None else str(raw_value),
                            "value": parsed_value,
                            "value_state": value_state,
                        }
                    )

    if not rows:
        return _empty_facts(), [*warnings, "sidra_nested_no_series_facts"]

    return pl.DataFrame(rows), warnings