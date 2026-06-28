from __future__ import annotations

from pegasus.datasus.declarative_normalize import (
    normalize_record,
    normalize_sim_do_record as _registry_normalize_sim_do_record,
    normalize_sinasc_record as _registry_normalize_sinasc_record,
)

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7


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
    "sex_state",
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


def _decode_sex(value: Any) -> tuple[str | None, str]:
    """Decode DATASUS SEXO via the shared single-authority decoder (§2.3)."""
    from pegasus.datasus.decoders import decode_datasus_sex

    decoded = decode_datasus_sex(value)
    return decoded.value, decoded.state


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


def _cod7_from_cod6(cod6: Any) -> str | None:
    """IBGE cod7 crosswalk of a decoded cod6 — a derivation, not a raw decode."""
    if not cod6:
        return None
    try:
        return datasus_cod6_to_ibge_cod7(str(cod6), strict=False)
    except Exception:
        return None


def _assemble_sim_do_record(
    registry_out: dict[str, Any],
    raw_row: dict[str, Any],
    *,
    idx: int,
    source_manifest_hash: str,
) -> dict[str, Any]:
    """Project registry-routed canonical fields into the SIM-DO output schema and
    compute the genuinely-derived fields that are not single-column decodes.

    The Source Field Registry (config/registries/source_fields.yaml) is the sole
    authority for raw→canonical routing and decoding (SHE-NORM-01 Path A): this
    assembler copies the registry's decode result and only adds identifiers,
    year, the age-provenance preference, the cod7 crosswalk, reporting delay, and
    hashes. It never re-decodes a raw column."""
    out: dict[str, Any] = {column: None for column in SIM_DO_NORMALIZED_COLUMNS}

    # 1. Registry-routed canonical values (the authoritative decode result).
    for column in SIM_DO_NORMALIZED_COLUMNS:
        if column in registry_out:
            out[column] = registry_out[column]

    # 2. Derived identifiers and provenance.
    out["event_id"] = f"sim_{idx}_{source_manifest_hash[:8]}"
    out["source_system"] = "SIM-DO"
    out["source_manifest_hash"] = source_manifest_hash
    out["raw_record_hash"] = _stable_hash(raw_row)
    out["processed_record_hash"] = _stable_hash({k: v for k, v in raw_row.items() if k != "raw_json"})

    # 3. Year is a derivation of the decoded death date (fallback to raw ANO).
    death_date = out.get("death_date")
    out["year"] = _year_from_date(death_date, raw_row.get("ANO"))

    # 4. Age: prefer the exact date difference when both dates resolved, else keep
    #    the registry decode_sim_idade result (MSD §2.4.1 age provenance).
    birth_date = out.get("birth_date")
    date_age_days = _days_between(birth_date, death_date)
    if date_age_days is not None and date_age_days >= 0:
        out["age_source"] = "date_difference"
        out["age_days"] = float(date_age_days)
        out["age_years"] = date_age_days / 365.25
    else:
        out["age_source"] = "IDADE"

    # 5. cod7 is the IBGE crosswalk of the decoded cod6 (a derivation, not a decode).
    out["mun_residence_cod7"] = _cod7_from_cod6(out.get("mun_residence_cod6"))
    out["mun_occurrence_cod7"] = _cod7_from_cod6(out.get("mun_occurrence_cod6"))

    # 6. Constant axis tag + cross-field reporting delay.
    out["race_axis_type"] = "administrative_death_declaration"
    out["reporting_delay"] = _days_between(death_date, out.get("certificate_date"))

    # 7. birth_weight is decoded as a float scalar; the schema column is integer grams.
    bw = out.get("birth_weight_death_context_grams")
    out["birth_weight_death_context_grams"] = int(bw) if bw is not None else None

    return out




def normalize_sim_do_events(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    """Registry-driven SIM-DO raw→canonical SHE normalizer (MSD §2.4.0/§2.4.1).

    SHE-NORM-01 Path A: this batch function is a thin wrapper over the single
    registry-driven record normalizer (`normalize_record`, via the Source Field
    Registry). The registry is the sole authority for raw→canonical routing and
    decoder dispatch; `_assemble_sim_do_record` only projects to the output
    schema and computes genuinely-derived fields (year, age provenance, cod7
    crosswalk, reporting delay, hashes). No raw column is decoded twice."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = _read_table(input_path)

    records = [
        _assemble_sim_do_record(
            normalize_record(row, source_system="SIM-DO"),
            row,
            idx=idx,
            source_manifest_hash=source_manifest_hash,
        )
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

