from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.hashing import sha256_json
from pegasus.datasus.parsers import (
    clean_scalar_string,
    normalize_municipality_code,
    parse_categorical,
    parse_datasus_age,
    parse_date,
    parse_icd10,
    parse_numeric,
)
from pegasus.datasus.schemas import SIM_DO_NORMALIZED_SCHEMA, enforce_schema


SOURCE_SYSTEM = "SIM-DO"

SIM_CAUSE_CHAIN_CANDIDATES = (
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    "CAUSABAS_O",
)

SIM_OPTIONAL_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "medical_assistance": ("ASSISTMED",),
    "death_during_pregnancy": ("GRAVIDEZ", "OBITOGRAV", "GESTACAO"),
    "death_during_puerperium": ("PUERPERIO", "OBITOPUER"),
    "death_type": ("TIPOBITO", "TIPOBITO1"),
    "gestational_weeks_death": ("SEMAGESTAC", "SEMAGEST", "GESTACAO"),
}


def normalize_sim_do(
    df: pl.DataFrame,
    *,
    source_manifest_hash: str = "",
    processed_record_hash_column: str | None = None,
) -> pl.DataFrame:
    """Normalize SIM-DO records into the PegaSUS event contract.

    This function accepts a raw-like or processed-like SIM dataframe, provided
    the canonical DATASUS columns are still present. It preserves invalid,
    missing, and unparseable states instead of forcing silent defaults.
    """
    rows: list[dict[str, Any]] = []

    columns_upper = {column.upper(): column for column in df.columns}

    for row_number, row in enumerate(df.iter_rows(named=True)):
        normalized = _normalize_sim_row(
            row,
            row_number=row_number,
            columns_upper=columns_upper,
            source_manifest_hash=source_manifest_hash,
            processed_record_hash_column=processed_record_hash_column,
        )
        rows.append(normalized)

    out = pl.DataFrame(rows) if rows else pl.DataFrame()
    return enforce_schema(out, SIM_DO_NORMALIZED_SCHEMA)


def _get(row: dict[str, Any], columns_upper: dict[str, str], name: str) -> Any:
    actual = columns_upper.get(name.upper())
    if actual is None:
        return None
    return row.get(actual)


def _get_first(row: dict[str, Any], columns_upper: dict[str, str], names: tuple[str, ...]) -> Any:
    for name in names:
        value = _get(row, columns_upper, name)
        if value is not None:
            return value
    return None


def _stateful_optional_categorical(
    row: dict[str, Any],
    columns_upper: dict[str, str],
    names: tuple[str, ...],
) -> tuple[str | None, str]:
    value = _get_first(row, columns_upper, names)
    parsed = parse_categorical(value)
    return _to_str_or_none(parsed.normalized), parsed.state


def _stateful_optional_numeric(
    row: dict[str, Any],
    columns_upper: dict[str, str],
    names: tuple[str, ...],
) -> tuple[float | None, str]:
    value = _get_first(row, columns_upper, names)
    parsed = parse_numeric(value)
    return _to_float_or_none(parsed.normalized), parsed.state


def _normalize_sim_row(
    row: dict[str, Any],
    *,
    row_number: int,
    columns_upper: dict[str, str],
    source_manifest_hash: str,
    processed_record_hash_column: str | None,
) -> dict[str, Any]:
    warnings: list[str] = []

    death_date = parse_date(_get(row, columns_upper, "DTOBITO"))
    birth_date = parse_date(_get(row, columns_upper, "DTNASC"))

    raw_age_value = _get(row, columns_upper, "IDADE")
    decoded_age = parse_datasus_age(raw_age_value)

    age_years, age_days, age_state, age_source, age_warnings = _resolve_age(
        death_date_iso=death_date.normalized,
        birth_date_iso=birth_date.normalized,
        decoded_age=decoded_age,
    )
    warnings.extend(age_warnings)

    sex = parse_categorical(
        _get(row, columns_upper, "SEXO"),
        valid_values={"0", "1", "2", "9", "M", "F", "I"},
        unknown_values={"0", "9", "I"},
    )

    race = parse_categorical(_get(row, columns_upper, "RACACOR"))
    race_missingness_state = race.state

    mun_res = normalize_municipality_code(_get(row, columns_upper, "CODMUNRES"))
    mun_ocor = normalize_municipality_code(_get(row, columns_upper, "CODMUNOCOR"))

    facility = parse_categorical(_get(row, columns_upper, "CODESTAB"))

    underlying_raw = _get(row, columns_upper, "CAUSABAS")
    underlying = parse_icd10(underlying_raw)

    cause_chain_raw, cause_chain_norm, cause_chain_states, cause_chain_warnings = (
        _parse_cause_chain(row, columns_upper)
    )
    warnings.extend(cause_chain_warnings)

    reporting_delay = parse_numeric(_get(row, columns_upper, "DIFDATA"))
    investigation_status = parse_categorical(_get(row, columns_upper, "TPPOS"))
    cause_altered = parse_categorical(_get(row, columns_upper, "ALTCAUSA"))

    medical_assistance, medical_assistance_state = _stateful_optional_categorical(
        row,
        columns_upper,
        SIM_OPTIONAL_FIELD_CANDIDATES["medical_assistance"],
    )
    death_during_pregnancy, death_during_pregnancy_state = _stateful_optional_categorical(
        row,
        columns_upper,
        SIM_OPTIONAL_FIELD_CANDIDATES["death_during_pregnancy"],
    )
    death_during_puerperium, death_during_puerperium_state = _stateful_optional_categorical(
        row,
        columns_upper,
        SIM_OPTIONAL_FIELD_CANDIDATES["death_during_puerperium"],
    )
    death_type, death_type_state = _stateful_optional_categorical(
        row,
        columns_upper,
        SIM_OPTIONAL_FIELD_CANDIDATES["death_type"],
    )
    gestational_weeks, gestational_weeks_state = _stateful_optional_numeric(
        row,
        columns_upper,
        SIM_OPTIONAL_FIELD_CANDIDATES["gestational_weeks_death"],
    )

    raw_record_hash = sha256_json(_json_safe_row(row))

    processed_record_hash = ""
    if processed_record_hash_column is not None:
        processed_hash_value = row.get(processed_record_hash_column)
        processed_record_hash = "" if processed_hash_value is None else str(processed_hash_value)

    return {
        "event_id": sha256_json(
            {
                "source_system": SOURCE_SYSTEM,
                "source_manifest_hash": source_manifest_hash,
                "row_number": row_number,
                "raw_record_hash": raw_record_hash,
            }
        ),
        "source_system": SOURCE_SYSTEM,
        "source_manifest_hash": source_manifest_hash,
        "row_number": row_number,

        "death_date": death_date.normalized,
        "death_date_state": death_date.state,
        "birth_date": birth_date.normalized,
        "birth_date_state": birth_date.state,

        "raw_age": _to_str_or_none(raw_age_value),
        "age_years": age_years,
        "age_days": age_days,
        "age_state": age_state,
        "age_source": age_source,

        "sex": _to_str_or_none(sex.normalized),
        "sex_state": sex.state,

        "race_color_admin": _to_str_or_none(race.normalized),
        "race_color_state": race.state,
        "race_axis_type": "administrative_death_declaration",
        "race_missingness_state": race_missingness_state,

        "mun_residence_cod6": mun_res.cod6,
        "mun_residence_cod7": mun_res.cod7,
        "mun_residence_state": mun_res.state,
        "mun_occurrence_cod6": mun_ocor.cod6,
        "mun_occurrence_cod7": mun_ocor.cod7,
        "mun_occurrence_state": mun_ocor.state,

        "facility_code": _to_str_or_none(facility.normalized),
        "facility_code_state": facility.state,

        "underlying_icd_raw": _to_str_or_none(underlying_raw),
        "underlying_icd_norm": underlying.normalized,
        "underlying_icd_parse_state": underlying.state,
        "underlying_icd_warnings": underlying.warnings,

        "cause_chain_raw": cause_chain_raw,
        "cause_chain_norm": cause_chain_norm,
        "cause_chain_parse_states": cause_chain_states,
        "cause_chain_warnings": cause_chain_warnings,

        "reporting_delay": _to_float_or_none(reporting_delay.normalized),
        "reporting_delay_state": reporting_delay.state,
        "investigation_status": _to_str_or_none(investigation_status.normalized),
        "investigation_status_state": investigation_status.state,
        "cause_altered": _to_str_or_none(cause_altered.normalized),
        "cause_altered_state": cause_altered.state,
        "medical_assistance": medical_assistance,
        "medical_assistance_state": medical_assistance_state,

        "death_during_pregnancy": death_during_pregnancy,
        "death_during_pregnancy_state": death_during_pregnancy_state,
        "death_during_puerperium": death_during_puerperium,
        "death_during_puerperium_state": death_during_puerperium_state,
        "death_type": death_type,
        "death_type_state": death_type_state,
        "gestational_weeks_death": gestational_weeks,
        "gestational_weeks_death_state": gestational_weeks_state,

        "raw_record_hash": raw_record_hash,
        "processed_record_hash": processed_record_hash,
        "normalization_warnings": warnings,
    }


def _resolve_age(
    *,
    death_date_iso: str | None,
    birth_date_iso: str | None,
    decoded_age,
) -> tuple[float | None, float | None, str, str, list[str]]:
    warnings: list[str] = []

    if death_date_iso is not None and birth_date_iso is not None:
        try:
            death = date.fromisoformat(death_date_iso)
            birth = date.fromisoformat(birth_date_iso)
            delta_days = (death - birth).days

            if delta_days < 0:
                warnings.append("birth_date_after_death_date")
            else:
                age_days = float(delta_days)
                age_years = age_days / 365.25
                return age_years, age_days, "valid", "date_difference", warnings
        except ValueError:
            warnings.append("age_date_difference_failed")

    if decoded_age.state == "valid":
        warnings.extend(decoded_age.warnings)
        return (
            decoded_age.age_years,
            decoded_age.age_days,
            decoded_age.state,
            decoded_age.source,
            warnings,
        )

    warnings.extend(decoded_age.warnings)
    return None, None, decoded_age.state, decoded_age.source, warnings


def _parse_cause_chain(
    row: dict[str, Any],
    columns_upper: dict[str, str],
) -> tuple[list[str], list[str], list[str], list[str]]:
    raw_values: list[str] = []
    normalized_values: list[str] = []
    states: list[str] = []
    warnings: list[str] = []

    for candidate in SIM_CAUSE_CHAIN_CANDIDATES:
        value = _get(row, columns_upper, candidate)
        text = clean_scalar_string(value)

        if text is None or text == "":
            continue

        parsed = parse_icd10(value)

        raw_values.append(text)
        if parsed.normalized is not None:
            normalized_values.append(parsed.normalized)
        states.append(parsed.state)
        warnings.extend([f"{candidate}:{warning}" for warning in parsed.warnings])

    return raw_values, normalized_values, states, warnings


def _to_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe_row(row: dict[str, Any]) -> dict[str, str | None]:
    safe: dict[str, str | None] = {}
    for key, value in row.items():
        if value is None:
            safe[key] = None
        else:
            try:
                json.dumps(value)
                safe[key] = str(value)
            except TypeError:
                safe[key] = repr(value)
    return safe