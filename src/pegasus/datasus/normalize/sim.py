from __future__ import annotations

from pegasus.datasus.normalize.records import (
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

from pegasus.datasus.normalize.completeness import check_raw_completeness
from pegasus.datasus.normalize.primitives import read_raw_table
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
    return read_raw_table(path)


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

    # 8. Categorical codebook translation (in-house replacement for microdatasus
    #    process_sim label decoding). Keyed off the raw column so the record oracle
    #    stays byte-aligned with the vectorized frame, which uses the same concepts.
    from pegasus.datasus.normalize.codebook import translate as _translate

    for field, (raw_col, concept) in _SIM_CATEGORICAL.items():
        out[field], _ = _translate(concept, raw_row.get(raw_col))

    return out


# output field → (raw column, codebook concept). Shared contract between the
# vectorized frame and the record oracle so both translate identically.
_SIM_CATEGORICAL: dict[str, tuple[str, str]] = {
    "place_of_death": ("LOCOCOR", "local_of_death"),
    "death_type": ("TIPOBITO", "death_type"),
    "fetal_or_liveborn_status_source": ("TIPOBITO", "death_type"),
    "maternal_education_legacy": ("ESCMAE", "education_years_sim"),
    "pregnancy_type": ("GRAVIDEZ", "pregnancy_type"),
    "gestational_age_group_death": ("GESTACAO", "gestation_group_sim"),
    "delivery_type_death_context": ("PARTO", "delivery_type"),
    "death_timing_relative_to_delivery": ("OBITOPARTO", "death_timing_delivery"),
    "death_during_pregnancy": ("OBITOGRAV", "yes_no"),
    "death_during_puerperium": ("OBITOPUERP", "puerperium_window"),
    "medical_assistance": ("ASSISTMED", "yes_no"),
    "exam_performed": ("EXAME", "yes_no"),
    "surgery_performed": ("CIRURGIA", "yes_no"),
    "autopsy_performed": ("NECROPSIA", "yes_no"),
    "investigation_status": ("TPPOS", "investigation_status"),
}




# SIM-DO IDADE unit scheme (code = zfill(3); unit = code//100, magnitude = code%100),
# mirroring decode_sim_idade: 1 hours, 2 days, 3 months, 4 years, 5 years_100_plus.
_SIM_AGE_UNITS = {
    1: (1 / (24 * 365.25), 1 / 24, 0.0, "hours"),
    2: (1 / 365.25, 1.0, 0.0, "days"),
    3: (1 / 12, 30.4375, 0.0, "months"),
    4: (1.0, 365.25, 0.0, "years"),
    5: (1.0, 365.25, 100.0, "years_100_plus"),
}
_SIM_CAUSE_CHAIN = (("A", "LINHAA"), ("B", "LINHAB"), ("C", "LINHAC"), ("D", "LINHAD"))


def _sim_vectorized_frame(df: pl.DataFrame, *, source_manifest_hash: str) -> pl.DataFrame:
    """Fully vectorized SIM-DO raw→canonical decode (MSD §2.4.0/§2.4.1).

    Reproduces the registry-routed record path (`normalize_record` +
    `_assemble_sim_do_record`) — the same raw→canonical field routing, decoders and
    derived fields (date-difference age provenance, cod7 crosswalk, reporting delay,
    content hashes) — via the shared `vec.Cols` primitives instead of a per-row
    Python loop. `normalize_sim_do_record` / `_assemble_sim_do_record` remain the
    record-level correctness oracle the equivalence stress-check pins against.
    """
    from pegasus.datasus.normalize.primitives import Cols, row_hash

    cx = Cols(df)
    crosswalk = _load_sim_crosswalk()
    raw_cols = [c for c in df.columns]

    def preserve(col: str) -> pl.Expr:
        return cx.clean(col)

    def cat(concept: str, col: str) -> pl.Expr:
        # Translate a coded categorical field via the in-house codebook registry
        # (single shared authority; replaces microdatasus process_* translation).
        return cx.categorical_value(concept, col)

    # -- dates & hour -----------------------------------------------------
    death_dt = cx.date("DTOBITO")
    birth_dt = cx.date("DTNASC")
    cert_dt = cx.date("DTATESTADO")
    invest_dt = cx.date("DTINVESTIG")
    death_iso = death_dt.cast(pl.Utf8)

    hd = cx.digits("HORAOBITO")
    hh = pl.when(hd.str.len_chars() <= 2).then(hd.cast(pl.Int64, strict=False)).otherwise(hd.str.zfill(4).str.slice(0, 2).cast(pl.Int64, strict=False))
    mm = pl.when(hd.str.len_chars() <= 2).then(pl.lit(0)).otherwise(hd.str.zfill(4).str.slice(2, 2).cast(pl.Int64, strict=False))
    hour_ok = hd.is_not_null() & (hh >= 0) & (hh <= 23) & (mm >= 0) & (mm <= 59)
    death_hour = pl.when(hour_ok).then(
        pl.concat_str([hh.cast(pl.Utf8).str.zfill(2), pl.lit(":"), mm.cast(pl.Utf8).str.zfill(2), pl.lit(":00")])
    ).otherwise(None)

    # -- SIM age (IDADE) + date-difference provenance ---------------------
    raw_age = cx.clean("IDADE")
    code = pl.when(raw_age.str.contains(r"^\d+$")).then(raw_age.str.zfill(3).cast(pl.Int64, strict=False)).otherwise(None)
    unit = code // 100
    mag = code % 100
    idade_years = pl.lit(None, dtype=pl.Float64)
    idade_days = pl.lit(None, dtype=pl.Float64)
    age_unit = pl.lit(None, dtype=pl.Utf8)
    for u, (yf, dfac, yoff, label) in _SIM_AGE_UNITS.items():
        idade_years = pl.when(unit == u).then(yoff + mag * yf).otherwise(idade_years)
        idade_days = pl.when(unit == u).then(365.25 * yoff + mag * dfac).otherwise(idade_days)
        age_unit = pl.when(unit == u).then(pl.lit(label)).otherwise(age_unit)

    diff_days = (death_dt - birth_dt).dt.total_days()
    has_diff = death_dt.is_not_null() & birth_dt.is_not_null() & (diff_days >= 0)
    age_source = pl.when(has_diff).then(pl.lit("date_difference")).otherwise(pl.lit("IDADE"))
    age_days = pl.when(has_diff).then(diff_days.cast(pl.Float64)).otherwise(idade_days)
    age_years = pl.when(has_diff).then(diff_days / 365.25).otherwise(idade_years)

    # -- municipality (cod6 + crosswalk cod7) -----------------------------
    res_cod6 = cx.municipality("r", "CODMUNRES", crosswalk=crosswalk)[0]
    occ_cod6 = cx.municipality("o", "CODMUNOCOR", crosswalk=crosswalk)[0]
    res_cod7 = res_cod6.replace_strict(crosswalk, default=None)
    occ_cod7 = occ_cod6.replace_strict(crosswalk, default=None)

    # -- facility code ----------------------------------------------------
    fd = cx.digits("CODESTAB")
    fac_allzero = fd.str.replace_all("0", "") == ""
    facility = pl.when(fd.is_null() | fac_allzero).then(None).otherwise(fd)
    facility_state = pl.when(cx.clean("CODESTAB").is_null() | fd.is_null() | fac_allzero).then(pl.lit("missing")).otherwise(pl.lit("valid"))

    # -- ICD (underlying + cause chain + associated) ----------------------
    underlying_raw = cx.first("CAUSABAS")
    underlying_norm = cx.icd_norm("CAUSABAS")
    underlying_state = cx.icd_state("CAUSABAS")

    # JSON objects formatted to match the record oracle's json.dumps(sort_keys=True)
    # output exactly: ", " between items and ": " after each key.
    def _kv(pos: str, present: pl.Expr, val: pl.Expr) -> pl.Expr:
        body = pl.when(val.is_null()).then(pl.lit("null")).otherwise(pl.lit('"') + val + pl.lit('"'))
        return pl.when(present).then(pl.lit(f'"{pos}": ') + body).otherwise(None)

    def _obj(parts: list[pl.Expr]) -> pl.Expr:
        return pl.lit("{") + pl.concat_str(parts, separator=", ", ignore_nulls=True) + pl.lit("}")

    chain_present = {col: cx.clean(col).is_not_null() for _, col in _SIM_CAUSE_CHAIN}
    cause_chain_raw = _obj([_kv(pos, chain_present[col], cx.clean(col)) for pos, col in _SIM_CAUSE_CHAIN])
    cause_chain_norm = _obj([_kv(pos, chain_present[col], cx.icd_norm(col)) for pos, col in _SIM_CAUSE_CHAIN])
    cause_chain_states = _obj([_kv(pos, chain_present[col], cx.icd_state(col)) for pos, col in _SIM_CAUSE_CHAIN])

    assoc_present = cx.clean("LINHAII").is_not_null()
    assoc_raw = cx.clean("LINHAII")
    assoc_norm = cx.icd_norm("LINHAII")
    assoc_state = pl.when(assoc_present).then(cx.icd_state("LINHAII")).otherwise(None)

    # -- scalars / counts / integers --------------------------------------
    def _integer(col: str) -> pl.Expr:
        f = cx.clean(col).str.replace_all(",", ".").cast(pl.Float64, strict=False)
        return pl.when(f == f.floor()).then(f.cast(pl.Int64)).otherwise(None)

    def _count2(col: str) -> pl.Expr:
        z = cx.clean(col).str.zfill(2)
        return pl.when(z == "99").then(None).when(z.str.contains(r"^\d{1,2}$")).then(z.cast(pl.Int64, strict=False)).otherwise(None)

    peso = cx.clean("PESO")
    peso_v = peso.str.replace_all(",", ".").cast(pl.Float64, strict=False)
    birth_weight = pl.when(peso.is_in(["0", "9999"]) | (peso_v < 300) | (peso_v > 7000)).then(None).otherwise(peso_v).cast(pl.Int64)

    sex_v, sex_s = cx.sex("SEXO")
    race_v, race_s = cx.race_admin("RACACOR")

    # -- derived: year, reporting delay, content hashes -------------------
    year = death_iso.str.slice(0, 4).cast(pl.Int64, strict=False)
    reporting_delay = (cert_dt - death_dt).dt.total_days()
    content = pl.concat_str([pl.col(c).cast(pl.Utf8).fill_null("") for c in raw_cols], separator="").hash().cast(pl.Utf8)

    out = df.with_row_index("_i").with_columns(
        pl.concat_str([pl.lit("sim_"), pl.col("_i").cast(pl.Utf8), pl.lit("_" + source_manifest_hash[:8])]).alias("event_id"),
        pl.lit("SIM-DO").alias("source_system"),
        year.alias("year"),
        death_iso.alias("death_date"),
        death_hour.alias("death_hour"),
        birth_dt.cast(pl.Utf8).alias("birth_date"),
        age_source.alias("age_source"),
        age_days.alias("age_days"),
        age_years.alias("age_years"),
        age_unit.alias("age_unit"),
        raw_age.alias("raw_age_code"),
        sex_v.alias("sex"), sex_s.alias("sex_state"),
        race_v.alias("race_color_admin"),
        pl.lit("administrative_death_declaration").alias("race_axis_type"),
        race_s.alias("race_missingness_state"),
        res_cod6.alias("mun_residence_cod6"), res_cod7.alias("mun_residence_cod7"),
        occ_cod6.alias("mun_occurrence_cod6"), occ_cod7.alias("mun_occurrence_cod7"),
        cat("local_of_death", "LOCOCOR").alias("place_of_death"),
        facility.alias("facility_code"), facility_state.alias("facility_code_state"),
        underlying_raw.alias("underlying_icd_raw"), underlying_norm.alias("underlying_icd_norm"), underlying_state.alias("underlying_icd_parse_state"),
        cause_chain_raw.alias("cause_chain_raw"), cause_chain_norm.alias("cause_chain_norm"), cause_chain_states.alias("cause_chain_parse_states"),
        assoc_raw.alias("associated_conditions_raw"), assoc_norm.alias("associated_conditions_norm"), assoc_state.alias("associated_conditions_parse_states"),
        cat("death_type", "TIPOBITO").alias("death_type"),
        cat("death_type", "TIPOBITO").alias("fetal_or_liveborn_status_source"),
        _integer("IDADEMAE").alias("maternal_age_years"),
        cat("education_years_sim", "ESCMAE").alias("maternal_education_legacy"),
        preserve("ESCMAE2010").alias("maternal_education_2010"),
        preserve("OCUPMAE").alias("maternal_occupation_cbo"),
        _count2("QTDFILVIVO").alias("maternal_living_children_count"),
        _count2("QTDFILMORT").alias("maternal_deceased_children_count"),
        cat("pregnancy_type", "GRAVIDEZ").alias("pregnancy_type"),
        _integer("SEMAGESTAC").alias("gestational_weeks_death"),
        cat("gestation_group_sim", "GESTACAO").alias("gestational_age_group_death"),
        cat("delivery_type", "PARTO").alias("delivery_type_death_context"),
        cat("death_timing_delivery", "OBITOPARTO").alias("death_timing_relative_to_delivery"),
        birth_weight.alias("birth_weight_death_context_grams"),
        cat("yes_no", "OBITOGRAV").alias("death_during_pregnancy"),
        cat("puerperium_window", "OBITOPUERP").alias("death_during_puerperium"),
        cat("yes_no", "ASSISTMED").alias("medical_assistance"),
        cat("yes_no", "EXAME").alias("exam_performed"),
        cat("yes_no", "CIRURGIA").alias("surgery_performed"),
        cat("yes_no", "NECROPSIA").alias("autopsy_performed"),
        preserve("COMUNSVOIM").alias("svo_iml_municipality"),
        cert_dt.cast(pl.Utf8).alias("certificate_date"),
        reporting_delay.alias("reporting_delay"),
        cat("investigation_status", "TPPOS").alias("investigation_status"),
        invest_dt.cast(pl.Utf8).alias("investigation_date"),
        preserve("CAUSABAS_O").alias("cause_altered"),
        content.alias("raw_record_hash"),
        content.alias("processed_record_hash"),
        pl.lit(source_manifest_hash).alias("source_manifest_hash"),
    )
    return out.select(SIM_DO_NORMALIZED_COLUMNS)


def _load_sim_crosswalk() -> dict[str, str]:
    from pegasus.geo.municipality_crosswalk import load_municipality_crosswalk

    return load_municipality_crosswalk()


def normalize_sim_do_events(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    """SIM-DO raw→canonical SHE normalizer (MSD §2.4.0/§2.4.1), fully vectorized.

    Uses `_sim_vectorized_frame` (shared `vec.Cols` primitives) instead of a per-row
    registry loop. The registry-routed record path (`normalize_record` +
    `_assemble_sim_do_record`) is retained as the correctness oracle the equivalence
    stress-check pins against; it is the same raw→canonical routing expressed as
    column operations."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = _read_table(input_path)
    missing = check_raw_completeness(df, "SIM-DO")
    out = _sim_vectorized_frame(df, source_manifest_hash=source_manifest_hash)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_count": out.height,
        "column_count": len(out.columns),
        "columns": out.columns,
        "missing_required_columns": missing,
    }

# ---- Hardline MSD SHE registry-routed entrypoint ----
def normalize_sim_do_record(row: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    registry_root = kwargs.get('registry_root', 'config/registries')
    return _registry_normalize_sim_do_record(row, registry_root=registry_root)

