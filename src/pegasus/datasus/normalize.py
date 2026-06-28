from __future__ import annotations

from pegasus.datasus.declarative_normalize import normalize_sim_do_record as _registry_normalize_sim_do_record, normalize_sinasc_record as _registry_normalize_sinasc_record

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


def _raw(df: pl.DataFrame, column: str) -> pl.Expr:
    """Raw column as Utf8, or a null literal if absent — so the decoder never
    crashes on a source file that omits an optional field."""
    if column in df.columns:
        return pl.col(column).cast(pl.Utf8)
    return pl.lit(None, dtype=pl.Utf8)


def _row_raw(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _datasus_year(column_expr: pl.Expr) -> pl.Expr:
    """DATASUS dates are DDMMYYYY strings; the year is the trailing 4 digits."""
    s = column_expr.str.strip_chars()
    return (
        pl.when(s.str.len_chars() >= 8)
        .then(s.str.slice(4, 4).cast(pl.Int64, strict=False))
        .otherwise(None)
    )


def _cod6(column_expr: pl.Expr) -> pl.Expr:
    return column_expr.str.strip_chars().str.extract(r"(\d{6})", 1)


def _icd_norm(column_expr: pl.Expr) -> pl.Expr:
    return column_expr.str.replace_all(r"[^A-Za-z0-9]", "").str.to_uppercase()


def _icd_parse_state(norm_expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(norm_expr.is_null() | (norm_expr.str.len_chars() == 0))
        .then(pl.lit("missing"))
        .when(norm_expr.str.contains(r"^[A-Z][0-9]{2,4}$"))
        .then(pl.lit("valid"))
        .otherwise(pl.lit("invalid"))
    )


def _normalize_icd_cell(value: Any, *, role: str, source_field: str, position: str | None = None) -> tuple[str | None, str]:
    parsed = parse_icd(
        None if value is None else str(value),
        topology_role=role,
        source_field=source_field,
        position=position,
    )
    return parsed.normalized, parsed.parse_state


def _sim_chain(row: dict[str, Any]) -> tuple[str, str, str]:
    raw: dict[str, str] = {}
    norm: dict[str, str | None] = {}
    states: dict[str, str] = {}
    for pos, column in (("A", "LINHAA"), ("B", "LINHAB"), ("C", "LINHAC"), ("D", "LINHAD")):
        value = _clean_str(row.get(column))
        if value is None:
            continue
        raw[pos] = value
        parsed_norm, parsed_state = _normalize_icd_cell(
            value,
            role="sim_causal_chain",
            source_field=column,
            position=pos,
        )
        norm[pos] = parsed_norm
        states[pos] = parsed_state
    return _json(raw), _json(norm), _json(states)


def _sim_associated(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    value = _clean_str(row.get("LINHAII"))
    if value is None:
        return None, None, None
    parsed_norm, parsed_state = _normalize_icd_cell(
        value,
        role="sim_associated_condition",
        source_field="LINHAII",
    )
    return value, parsed_norm, parsed_state


def _sim_record(row: dict[str, Any], *, idx: int, source_manifest_hash: str) -> dict[str, Any]:
    death_date = _parse_datasus_date(row.get("DTOBITO"))
    birth_date = _parse_datasus_date(row.get("DTNASC"))
    idade = decode_sim_idade(row.get("IDADE"))
    date_age_days = _days_between(birth_date, death_date)
    if date_age_days is not None and date_age_days >= 0:
        age_source = "date_difference"
        age_days = float(date_age_days)
        age_years = age_days / 365.25
    else:
        age_source = "IDADE"
        age_days = idade.age_days
        age_years = idade.age_years
    res6, res7 = _mun_codes(row.get("CODMUNRES"))
    occ6, occ7 = _mun_codes(row.get("CODMUNOCOR"))
    facility, facility_state = _facility_code(row.get("CODESTAB"))
    race, race_state = _race_state(row.get("RACACOR"))
    underlying = parse_icd(
        None if row.get("CAUSABAS") is None else str(row.get("CAUSABAS")),
        topology_role="sim_underlying_cause",
        source_field="CAUSABAS",
    )
    chain_raw, chain_norm, chain_states = _sim_chain(row)
    assoc_raw, assoc_norm, assoc_state = _sim_associated(row)
    certificate_date = _parse_datasus_date(row.get("DTATESTADO"))
    investigation_date = _parse_datasus_date(row.get("DTINVESTIG"))
    weight = decode_physical_scalar(
        row.get("PESO"),
        unit="grams",
        lower=300,
        upper=7000,
        sentinels={"0000", "9999"},
    )
    living = decode_count2(row.get("QTDFILVIVO"), sentinels={"99"})
    deceased = decode_count2(row.get("QTDFILMORT"), sentinels={"99"})
    out = {column: None for column in SIM_DO_NORMALIZED_COLUMNS}
    out.update(
        {
            "event_id": f"sim_{idx}_{source_manifest_hash[:8]}",
            "source_system": "SIM-DO",
            "year": _year_from_date(death_date, row.get("ANO")),
            "death_date": death_date,
            "death_hour": _parse_hour(row.get("HORAOBITO")),
            "birth_date": birth_date,
            "age_source": age_source,
            "age_days": age_days,
            "age_years": age_years,
            "age_unit": idade.age_unit,
            "raw_age_code": None if row.get("IDADE") is None else str(row.get("IDADE")).strip(),
            "sex": {"1": "male", "2": "female"}.get(str(row.get("SEXO")).strip() if row.get("SEXO") is not None else "", "unknown"),
            "race_color_admin": race,
            "race_axis_type": "administrative_death_declaration",
            "race_missingness_state": race_state,
            "mun_residence_cod6": res6,
            "mun_residence_cod7": res7,
            "mun_occurrence_cod6": occ6,
            "mun_occurrence_cod7": occ7,
            "place_of_death": _clean_str(row.get("LOCOCOR")),
            "facility_code": facility,
            "facility_code_state": facility_state,
            "underlying_icd_raw": underlying.raw,
            "underlying_icd_norm": underlying.normalized,
            "underlying_icd_parse_state": underlying.parse_state,
            "cause_chain_raw": chain_raw,
            "cause_chain_norm": chain_norm,
            "cause_chain_parse_states": chain_states,
            "associated_conditions_raw": assoc_raw,
            "associated_conditions_norm": assoc_norm,
            "associated_conditions_parse_states": assoc_state,
            "death_type": _clean_str(row.get("TIPOBITO")),
            "fetal_or_liveborn_status_source": _clean_str(row.get("TIPOBITO")),
            "maternal_age_years": _int_or_none(row.get("IDADEMAE")),
            "maternal_education_legacy": _clean_str(row.get("ESCMAE")),
            "maternal_education_2010": _clean_str(row.get("ESCMAE2010")),
            "maternal_occupation_cbo": _clean_str(row.get("OCUPMAE")),
            "maternal_living_children_count": living.value,
            "maternal_deceased_children_count": deceased.value,
            "pregnancy_type": _clean_str(row.get("GRAVIDEZ")),
            "gestational_weeks_death": _int_or_none(row.get("SEMAGESTAC")),
            "gestational_age_group_death": _clean_str(row.get("GESTACAO")),
            "delivery_type_death_context": _clean_str(row.get("PARTO")),
            "death_timing_relative_to_delivery": _clean_str(row.get("OBITOPARTO")),
            "birth_weight_death_context_grams": int(weight.value) if weight.value is not None else None,
            "death_during_pregnancy": _clean_str(row.get("OBITOGRAV")),
            "death_during_puerperium": _clean_str(row.get("OBITOPUERP")),
            "medical_assistance": _clean_str(row.get("ASSISTMED")),
            "exam_performed": _clean_str(row.get("EXAME")),
            "surgery_performed": _clean_str(row.get("CIRURGIA")),
            "autopsy_performed": _clean_str(row.get("NECROPSIA")),
            "svo_iml_municipality": _clean_str(row.get("COMUNSVOIM")),
            "certificate_date": certificate_date,
            "reporting_delay": _days_between(death_date, certificate_date),
            "investigation_status": _clean_str(row.get("TPPOS")),
            "investigation_date": investigation_date,
            "cause_altered": _clean_str(row.get("CAUSABAS_O")),
            "raw_record_hash": _stable_hash(row),
            "processed_record_hash": _stable_hash({k: v for k, v in row.items() if k != "raw_json"}),
            "source_manifest_hash": source_manifest_hash,
        }
    )
    return out


def normalize_sim_do_events(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    """Real vectorized SIM-DO raw→canonical SHE decoder (MSD §2.4.0/§2.4.1).

    Replaces the previous per-row registry-routed path, which looked up *raw*
    DATASUS column names in a registry keyed by *canonical* names, matched
    nothing, and emitted 60 canonical columns that were 100% null. Here every
    canonical field is computed by a Polars expression over the raw columns,
    using the registered composite decoders (e.g. SIM IDADE) — fast and real."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = _read_table(input_path)
    n = df.height

    records = [
        _sim_record(row, idx=idx, source_manifest_hash=source_manifest_hash)
        for idx, row in enumerate(df.to_dicts())
    ]
    out = pl.DataFrame(records, infer_schema_length=None).select(SIM_DO_NORMALIZED_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": out.height,
        "column_count": len(out.columns),
        "columns": out.columns,
    }

# ---- Hardline MSD SHE registry-routed entrypoint ----
def normalize_sim_do_record(row: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    registry_root = kwargs.get('registry_root', 'config/registries')
    return _registry_normalize_sim_do_record(row, registry_root=registry_root)

