from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.datasus.decoders import (
    decode_count2,
    decode_physical_scalar,
    decode_sim_idade,
)
from pegasus.datasus.icd_parser import parse_icd


SIM_DO_NORMALIZED_COLUMNS = [
    "event_id",
    "source_system",
    "year",
    "death_date",
    "death_hour",
    "birth_date",
    "age_source",
    "age_days",
    "age_years",
    "age_unit",
    "raw_age_code",
    "sex",
    "race_color_admin",
    "race_axis_type",
    "race_missingness_state",
    "mun_residence_cod6",
    "mun_residence_cod7",
    "mun_occurrence_cod6",
    "mun_occurrence_cod7",
    "place_of_death",
    "facility_code",
    "facility_code_state",
    "underlying_icd_raw",
    "underlying_icd_norm",
    "underlying_icd_parse_state",
    "cause_chain_raw",
    "cause_chain_norm",
    "cause_chain_parse_states",
    "associated_conditions_raw",
    "associated_conditions_norm",
    "associated_conditions_parse_states",
    "death_type",
    "fetal_or_liveborn_status_source",
    "maternal_age_years",
    "maternal_education_legacy",
    "maternal_education_2010",
    "maternal_occupation_cbo",
    "maternal_living_children_count",
    "maternal_deceased_children_count",
    "pregnancy_type",
    "gestational_weeks_death",
    "gestational_age_group_death",
    "delivery_type_death_context",
    "death_timing_relative_to_delivery",
    "birth_weight_death_context_grams",
    "death_during_pregnancy",
    "death_during_puerperium",
    "medical_assistance",
    "exam_performed",
    "surgery_performed",
    "autopsy_performed",
    "svo_iml_municipality",
    "certificate_date",
    "reporting_delay",
    "investigation_status",
    "investigation_date",
    "cause_altered",
    "raw_record_hash",
    "processed_record_hash",
    "source_manifest_hash",
]


INVALID_DATE_VALUES = {
    "",
    "0",
    "00000000",
    "0000-00-00",
    "00/00/0000",
    "99999999",
    "9999-99-99",
    "99/99/9999",
    "NA",
    "NAN",
    "NULL",
}


def _raw(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NULL"}:
        return None
    return text


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(path, infer_schema_length=1000, ignore_errors=False)
    if suffix in {".json", ".ndjson"}:
        return pl.read_ndjson(path)

    raise ValueError(f"Unsupported SIM-DO input format: {path}")


def _parse_datasus_date(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None

    upper = text.upper()
    if upper in INVALID_DATE_VALUES:
        return None

    # Field-specific date parser. Do not apply this generically to identifiers.
    candidates = [
        "%d%m%Y",
        "%Y%m%d",
        "%d/%m/%Y",
        "%Y-%m-%d",
    ]

    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    return None


def _parse_hour(value: Any) -> str | None:
    text = _clean_str(value)
    if text is None:
        return None

    digits = re.sub(r"\D", "", text)
    if digits == "":
        return None

    if len(digits) <= 2:
        hour = int(digits)
        minute = 0
    else:
        padded = digits.zfill(4)
        hour = int(padded[:2])
        minute = int(padded[2:4])

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    return f"{hour:02d}:{minute:02d}:00"


def _year_from_date(iso_date: str | None, fallback: Any = None) -> int | None:
    if iso_date:
        return int(iso_date[:4])
    if fallback is not None:
        text = _clean_str(fallback)
        if text and text.isdigit() and len(text) == 4:
            return int(text)
    return None


def _mun_codes(value: Any) -> tuple[str | None, str | None]:
    text = _clean_str(value)
    if text is None:
        return None, None

    digits = re.sub(r"\D", "", text)
    if len(digits) == 6:
        return digits, None
    if len(digits) == 7:
        return digits[:6], digits

    return None, None


def _facility_code(value: Any) -> tuple[str | None, str]:
    text = _clean_str(value)
    if text is None:
        return None, "missing"
    digits = re.sub(r"\D", "", text)
    if digits == "" or set(digits) == {"0"}:
        return None, "missing"
    return digits, "valid"


def _race_state(value: Any) -> tuple[str | None, str]:
    text = _clean_str(value)
    if text is None:
        return None, "missing"

    if text in {"9", "99"}:
        return text, "unknown"

    return text, "valid"


def _int_or_none(value: Any) -> int | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return int(float(text.replace(",", ".")))
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    text = _clean_str(value)
    if text is None:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _days_between(start_iso: str | None, end_iso: str | None) -> int | None:
    if not start_iso or not end_iso:
        return None
    try:
        start = datetime.strptime(start_iso, "%Y-%m-%d").date()
        end = datetime.strptime(end_iso, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (end - start).days


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def normalize_sim_do_record(row: dict[str, Any], *, source_manifest_hash: str) -> dict[str, Any]:
    death_date = _parse_datasus_date(_raw(row, "DTOBITO", "death_date"))
    birth_date = _parse_datasus_date(_raw(row, "DTNASC", "birth_date"))
    death_hour = _parse_hour(_raw(row, "HORAOBITO", "death_hour"))
    year = _year_from_date(death_date, _raw(row, "ANO", "year"))

    decoded_age = decode_sim_idade(_raw(row, "IDADE", "age_source"))
    date_age_days = _days_between(birth_date, death_date)

    age_days = float(date_age_days) if date_age_days is not None and date_age_days >= 0 else decoded_age.age_days
    age_years = (age_days / 365.25) if age_days is not None else decoded_age.age_years
    age_source = "date_difference" if date_age_days is not None and date_age_days >= 0 else "IDADE"

    race, race_state = _race_state(_raw(row, "RACACOR", "race_color_admin"))
    mun_res6, mun_res7 = _mun_codes(_raw(row, "CODMUNRES", "mun_residence"))
    mun_occ6, mun_occ7 = _mun_codes(_raw(row, "CODMUNOCOR", "mun_occurrence"))
    facility_code, facility_state = _facility_code(_raw(row, "CODESTAB", "facility_code"))

    underlying = parse_icd(
        _raw(row, "CAUSABAS", "underlying_icd_raw"),
        topology_role="underlying_cause",
        source_field="CAUSABAS",
    )

    chain_fields = [
        ("LINHAA", "A"),
        ("LINHAB", "B"),
        ("LINHAC", "C"),
        ("LINHAD", "D"),
    ]
    chain_raw: dict[str, Any] = {}
    chain_norm: dict[str, Any] = {}
    chain_states: dict[str, Any] = {}

    for field, position in chain_fields:
        raw_value = _raw(row, field)
        parsed = parse_icd(
            raw_value,
            topology_role="terminal_chain",
            source_field=field,
            position=position,
        )
        chain_raw[position] = None if raw_value is None else str(raw_value)
        chain_norm[position] = parsed.normalized
        chain_states[position] = parsed.parse_state

    associated_raw = _raw(row, "LINHAII")
    associated = parse_icd(
        associated_raw,
        topology_role="associated_condition",
        source_field="LINHAII",
    )

    living_children = decode_count2(_raw(row, "QTDFILVIVO"), sentinels={"99"})
    deceased_children = decode_count2(_raw(row, "QTDFILMORT"), sentinels={"99"})

    birth_weight = decode_physical_scalar(
        _raw(row, "PESO"),
        unit="grams",
        lower=300,
        upper=6500,
        sentinels={"9999", "0000"},
    )

    certificate_date = _parse_datasus_date(_raw(row, "DTATESTADO"))
    investigation_date = _parse_datasus_date(_raw(row, "DTINVESTIG"))

    reporting_delay = _days_between(certificate_date, death_date)
    if reporting_delay is not None:
        reporting_delay = abs(reporting_delay)

    raw_hash = _stable_hash(row)

    normalized = {
        "event_id": _stable_hash(
            {
                "source": "SIM-DO",
                "death_date": death_date,
                "mun_residence": mun_res6,
                "underlying": underlying.normalized,
                "raw_hash": raw_hash,
            }
        ),
        "source_system": "SIM-DO",
        "year": year,
        "death_date": death_date,
        "death_hour": death_hour,
        "birth_date": birth_date,
        "age_source": age_source,
        "age_days": age_days,
        "age_years": age_years,
        "age_unit": decoded_age.age_unit,
        "raw_age_code": None if _raw(row, "IDADE") is None else str(_raw(row, "IDADE")).strip(),
        "sex": _clean_str(_raw(row, "SEXO", "sex")),
        "race_color_admin": race,
        "race_axis_type": "administrative_death_declaration",
        "race_missingness_state": race_state,
        "mun_residence_cod6": mun_res6,
        "mun_residence_cod7": mun_res7,
        "mun_occurrence_cod6": mun_occ6,
        "mun_occurrence_cod7": mun_occ7,
        "place_of_death": _clean_str(_raw(row, "LOCOCOR")),
        "facility_code": facility_code,
        "facility_code_state": facility_state,
        "underlying_icd_raw": underlying.raw,
        "underlying_icd_norm": underlying.normalized,
        "underlying_icd_parse_state": underlying.parse_state,
        "cause_chain_raw": _json(chain_raw),
        "cause_chain_norm": _json(chain_norm),
        "cause_chain_parse_states": _json(chain_states),
        "associated_conditions_raw": None if associated_raw is None else str(associated_raw),
        "associated_conditions_norm": associated.normalized,
        "associated_conditions_parse_states": associated.parse_state,
        "death_type": _clean_str(_raw(row, "TIPOBITO", "death_type")),
        "fetal_or_liveborn_status_source": _clean_str(_raw(row, "TIPOBITO", "OBITOFETAL", "death_type")),
        "maternal_age_years": _int_or_none(_raw(row, "IDADEMAE")),
        "maternal_education_legacy": _clean_str(_raw(row, "ESCMAE")),
        "maternal_education_2010": _clean_str(_raw(row, "ESCMAE2010")),
        "maternal_occupation_cbo": _clean_str(_raw(row, "OCUPMAE")),
        "maternal_living_children_count": living_children.value,
        "maternal_deceased_children_count": deceased_children.value,
        "pregnancy_type": _clean_str(_raw(row, "GRAVIDEZ")),
        "gestational_weeks_death": _int_or_none(_raw(row, "SEMAGESTAC")),
        "gestational_age_group_death": _clean_str(_raw(row, "GESTACAO")),
        "delivery_type_death_context": _clean_str(_raw(row, "PARTO")),
        "death_timing_relative_to_delivery": _clean_str(_raw(row, "OBITOPARTO")),
        "birth_weight_death_context_grams": birth_weight.value,
        "death_during_pregnancy": _clean_str(_raw(row, "OBITOGRAV")),
        "death_during_puerperium": _clean_str(_raw(row, "OBITOPUERP")),
        "medical_assistance": _clean_str(_raw(row, "ASSISTMED")),
        "exam_performed": _clean_str(_raw(row, "EXAME")),
        "surgery_performed": _clean_str(_raw(row, "CIRURGIA")),
        "autopsy_performed": _clean_str(_raw(row, "NECROPSIA")),
        "svo_iml_municipality": _clean_str(_raw(row, "COMUNSVOIM")),
        "certificate_date": certificate_date,
        "reporting_delay": reporting_delay,
        "investigation_status": _clean_str(_raw(row, "TPPOS")),
        "investigation_date": investigation_date,
        "cause_altered": _clean_str(_raw(row, "CAUSABAS_O", "ALTERADA", "cause_altered")),
        "raw_record_hash": raw_hash,
        "processed_record_hash": "",
        "source_manifest_hash": source_manifest_hash,
    }

    normalized["processed_record_hash"] = _stable_hash({k: v for k, v in normalized.items() if k != "processed_record_hash"})
    return normalized


def normalize_sim_do_events(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)

    raw_df = _read_table(input_path)
    rows = raw_df.to_dicts()

    normalized_rows = [
        normalize_sim_do_record(row, source_manifest_hash=source_manifest_hash)
        for row in rows
    ]

    df = pl.DataFrame(normalized_rows, infer_schema_length=None)

    for column in SIM_DO_NORMALIZED_COLUMNS:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(column))

    df = df.select(SIM_DO_NORMALIZED_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": df.height,
        "column_count": len(df.columns),
        "columns": df.columns,
    }
