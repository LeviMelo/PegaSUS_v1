from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.datasus.decoders import decode_datasus_sex, decode_sih_age, filter_cnpj
from pegasus.datasus.icd_parser import parse_icd
from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7

SECONDARY_DIAG_COLUMNS = tuple(f"DIAGSEC{i}" for i in range(1, 10))
SECONDARY_TYPE_COLUMNS = tuple(f"TPDISEC{i}" for i in range(1, 10))
COST_COMPONENTS = ("VAL_SH", "VAL_SP", "VAL_UTI", "VAL_TOT")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return None
    return text


def _digits(value: Any) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    digits = re.sub(r"\D", "", text)
    return digits or None


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pl.read_csv(path, infer_schema_length=0, ignore_errors=False)
    raise ValueError(f"Unsupported SIH-RD input format: {path}")


def _parse_date(value: Any) -> tuple[str | None, int | None, str]:
    text = _clean(value)
    if text is None:
        return None, None, "missing"
    digits = _digits(text)
    candidates: list[str] = []
    if digits and len(digits) == 8:
        candidates.append(f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}")
        candidates.append(f"{digits[4:8]}-{digits[2:4]}-{digits[0:2]}")
    candidates.append(text)
    for candidate in candidates:
        try:
            d = datetime.fromisoformat(candidate).date()
            return d.isoformat(), d.year, "valid"
        except Exception:
            pass
    return None, None, "invalid"


def _mun(value: Any) -> tuple[str | None, str | None, str]:
    digits = _digits(value)
    if digits is None:
        return None, None, "missing"
    cod6 = digits if len(digits) == 6 else (digits[:6] if len(digits) == 7 else None)
    if cod6 is None:
        return None, None, "invalid"
    # DATASUS "município ignorado" sentinel (UF + 0000) is missingness, not geography.
    if cod6[2:] == "0000":
        return None, None, "ignored_municipality"
    if len(digits) == 7:
        return cod6, digits, "ibge_cod7"
    try:
        return cod6, datasus_cod6_to_ibge_cod7(cod6, strict=True), "datasus_cod6"
    except Exception:
        return cod6, None, "unmapped_datasus_cod6"


def _number(value: Any) -> tuple[float | None, str, str | None]:
    raw = _clean(value)
    if raw is None:
        return None, "missing", None
    try:
        parsed = float(raw.replace(",", "."))
    except ValueError:
        return None, "invalid", raw
    return (parsed, "valid", raw) if parsed >= 0 else (None, "invalid", raw)


def _int_nonnegative(value: Any) -> tuple[int | None, str]:
    number, state, _ = _number(value)
    return (int(number), state) if number is not None else (None, state)


def _decode_sex(value: Any) -> tuple[str | None, str]:
    """Decode DATASUS SEXO via the shared single-authority decoder (§2.3)."""
    decoded = decode_datasus_sex(value)
    return decoded.value, decoded.state


def _death_flag(value: Any) -> tuple[bool | None, str]:
    text = _clean(value)
    if text is None:
        return None, "missing"
    norm = text.casefold()
    if norm in {"1", "sim", "yes", "true", "t", "morte", "death"}:
        return True, "valid"
    if norm in {"0", "não", "nao", "no", "false", "f"}:
        return False, "valid"
    return None, "invalid"


def normalize_sih_rd_record(row: dict[str, Any], *, source_manifest_hash: str) -> dict[str, Any]:
    raw_payload = {str(k): v for k, v in row.items()}
    admission_date, admission_year, admission_state = _parse_date(row.get("DT_INTER") or row.get("DTINTERN") or row.get("admission_date"))
    discharge_date, _, discharge_state = _parse_date(row.get("DT_SAIDA") or row.get("DTSAIDA") or row.get("discharge_date"))
    cod6, cod7, mun_state = _mun(row.get("MUNIC_RES") or row.get("CODMUNRES") or row.get("municipality"))
    age = decode_sih_age(row.get("COD_IDADE") or row.get("CODIDADE") or "4", row.get("IDADE") or row.get("age"))
    principal = parse_icd(row.get("DIAG_PRINC") or row.get("principal_icd"), topology_role="sih_principal_diagnosis", source_field="DIAG_PRINC")
    secondary_raw: dict[str, Any] = {}
    secondary_norm: dict[str, Any] = {}
    secondary_states: dict[str, Any] = {}
    for col in SECONDARY_DIAG_COLUMNS:
        if col in row:
            parsed = parse_icd(row.get(col), topology_role="sih_secondary_diagnosis", source_field=col)
            secondary_raw[col] = None if row.get(col) is None else str(row.get(col))
            secondary_norm[col] = parsed.normalized
            secondary_states[col] = parsed.parse_state
    secondary_type = {col: _clean(row.get(col)) for col in SECONDARY_TYPE_COLUMNS if col in row}
    stay_days, stay_state = _int_nonnegative(row.get("DIAS_PERM") or row.get("QT_DIARIAS") or row.get("stay_length_days"))
    icu_days_month, icu_month_state = _int_nonnegative(row.get("UTI_MES_TO"))
    icu_days_adm, icu_adm_state = _int_nonnegative(row.get("UTI_INT_TO"))
    death, death_state = _death_flag(row.get("MORTE") or row.get("OBITO"))
    costs: dict[str, float | None] = {}
    cost_states: dict[str, str] = {}
    cost_raw: dict[str, str | None] = {}
    for col in COST_COMPONENTS:
        value, state, raw = _number(row.get(col))
        costs[col] = value
        cost_states[col] = state
        cost_raw[col] = raw
    sex_value, sex_st = _decode_sex(row.get("SEXO"))
    # Movement geography (MUNIC_MOV) — where the admission was processed; distinct
    # from residence (MUNIC_RES). Required by the §5.9 FacilityFlow bridge.
    mov6, mov7, mov_state = _mun(row.get("MUNIC_MOV") or row.get("MUNIC_MOVI") or row.get("mun_movement"))
    # Hospital corporate linkage (CGC_HOSP) — §2.4.0.5 CNPJ gate; required by the
    # FacilityFlow many-to-many guard.
    hospital_cnpj = filter_cnpj(row.get("CGC_HOSP") or row.get("hospital_cnpj"))
    facility_cnes = _clean(row.get("CNES") or row.get("facility_code"))
    manager_cnpj = filter_cnpj(row.get("CNPJ_MANT") or row.get("GESTOR_CPF") or row.get("maintainer_cnpj"))
    admission_id = _clean(row.get("AIH") or row.get("N_AIH") or row.get("admission_id")) or _stable_hash(raw_payload)[:16]
    record_state = "valid"
    if admission_year is None or cod6 is None or principal.parse_state in {"missing", "blank", "unparseable", "invalid"}:
        record_state = "invalid_identity_or_principal_diagnosis"
    return {
        "admission_id": f"SIH-{admission_id}",
        "source_system": "SIH-RD",
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "admission_year": admission_year,
        "admission_date_state": admission_state,
        "discharge_date_state": discharge_state,
        "mun_residence_cod6": cod6,
        "mun_residence_cod7": cod7,
        "municipality_code_state": mun_state,
        "mun_movement_cod6": mov6,
        "mun_movement_cod7": mov7,
        "mun_movement_state": mov_state,
        "facility_cnes": facility_cnes,
        "hospital_cnpj": hospital_cnpj.cnpj,
        "hospital_cnpj_state": hospital_cnpj.state,
        "maintainer_cnpj": manager_cnpj.cnpj,
        "maintainer_cnpj_state": manager_cnpj.state,
        "age_days": age.age_days,
        "age_years": age.age_years,
        "age_unit": age.age_unit,
        "age_state": age.state,
        "sex": sex_value,
        "sex_state": sex_st,
        "race_color_billing": _clean(row.get("RACA_COR")),
        "race_axis_type": "sih_billing_race_color",
        "principal_icd_raw": principal.raw,
        "principal_icd_norm": principal.normalized,
        "principal_icd_parse_state": principal.parse_state,
        "secondary_icd_raw_json": json.dumps(secondary_raw, ensure_ascii=False, sort_keys=True),
        "secondary_icd_norm_json": json.dumps(secondary_norm, ensure_ascii=False, sort_keys=True),
        "secondary_icd_parse_states_json": json.dumps(secondary_states, ensure_ascii=False, sort_keys=True),
        "secondary_diagnosis_type_json": json.dumps(secondary_type, ensure_ascii=False, sort_keys=True),
        "procedure_requested": _clean(row.get("PROC_SOLIC")),
        "procedure_performed": _clean(row.get("PROC_REA")),
        "stay_length_days": stay_days,
        "stay_length_state": stay_state,
        "icu_type_mark": _clean(row.get("MARCA_UTI")),
        "icu_days_month_total": icu_days_month,
        "icu_days_month_state": icu_month_state,
        "icu_days_hospitalization_total": icu_days_adm,
        "icu_days_hospitalization_state": icu_adm_state,
        "death_flag": death,
        "death_flag_state": death_state,
        "hospital_service_cost_real": costs["VAL_SH"],
        "professional_service_cost_real": costs["VAL_SP"],
        "icu_cost_real": costs["VAL_UTI"],
        "total_admission_cost_real": costs["VAL_TOT"],
        "cost_state_json": json.dumps(cost_states, ensure_ascii=False, sort_keys=True),
        "cost_raw_json": json.dumps(cost_raw, ensure_ascii=False, sort_keys=True),
        "record_state": record_state,
        "source_manifest_hash": source_manifest_hash,
        "row_hash": _stable_hash(raw_payload),
        "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str),
    }


def normalize_sih_rd_events(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, Any]:
    """Real vectorized SIH-RD raw→canonical SHE decoder (MSD §2.4.2).

    Hospital admissions: competence year, residence cod6, principal diagnosis,
    in-hospital death flag, length of stay, and the four distinct economic cost
    components (VAL_SH/SP/UTI/TOT) kept separate per §2.4.2 (not pooled)."""
    df = _read_table(input_path).with_row_index("_row_idx")
    def col(name: str) -> pl.Expr:
        return pl.col(name) if name in df.columns else pl.lit(None)

    def first(*names: str) -> pl.Expr:
        return pl.coalesce([col(name) for name in names])

    def clean_expr(expr: pl.Expr) -> pl.Expr:
        text = expr.cast(pl.Utf8).str.strip_chars()
        return pl.when(text.str.to_uppercase().is_in(["", "NA", "NAN", "NULL", "NONE"])).then(None).otherwise(text)

    def clean(*names: str) -> pl.Expr:
        return clean_expr(first(*names))

    def digits_expr(expr: pl.Expr) -> pl.Expr:
        d = clean_expr(expr).str.replace_all(r"\D", "")
        return pl.when(d == "").then(None).otherwise(d)

    def digits(*names: str) -> pl.Expr:
        return digits_expr(first(*names))

    def parsed_date(*names: str) -> pl.Expr:
        d = digits(*names)
        ymd = d.str.strptime(pl.Date, "%Y%m%d", strict=False)
        dmy = d.str.strptime(pl.Date, "%d%m%Y", strict=False)
        return pl.coalesce([ymd, dmy])

    def municipality(prefix: str, *names: str) -> list[pl.Expr]:
        d = digits(*names)
        cod6 = (
            pl.when(d.str.len_chars() == 6).then(d)
            .when(d.str.len_chars() == 7).then(d.str.slice(0, 6))
            .otherwise(None)
        )
        ignored = cod6.str.slice(2, 4) == "0000"
        valid = cod6.is_not_null() & ~ignored
        return [
            pl.when(valid).then(cod6).otherwise(None).alias(f"{prefix}_cod6"),
            pl.when(valid & (d.str.len_chars() == 7)).then(d).otherwise(None).alias(f"{prefix}_cod7"),
            pl.when(d.is_null()).then(pl.lit("missing"))
            .when(ignored).then(pl.lit("ignored_municipality"))
            .when(d.str.len_chars() == 7).then(pl.lit("ibge_cod7"))
            .when(d.str.len_chars() == 6).then(pl.lit("datasus_cod6"))
            .otherwise(pl.lit("invalid"))
            .alias(f"{prefix}_state"),
        ]

    def nonnegative_int(out_name: str, *names: str) -> list[pl.Expr]:
        raw = clean(*names)
        value = raw.str.replace_all(",", ".").cast(pl.Float64, strict=False)
        valid = value.is_not_null() & (value >= 0)
        return [
            pl.when(valid).then(value.cast(pl.Int64)).otherwise(None).alias(out_name),
            pl.when(raw.is_null()).then(pl.lit("missing"))
            .when(valid).then(pl.lit("valid"))
            .otherwise(pl.lit("invalid")).alias(f"{out_name}_state"),
        ]

    def money(out_name: str, *names: str) -> pl.Expr:
        value = clean(*names).str.replace_all(",", ".").cast(pl.Float64, strict=False)
        return pl.when(value >= 0).then(value).otherwise(None).alias(out_name)

    def cnpj_state(prefix: str, *names: str) -> list[pl.Expr]:
        raw = clean(*names)
        return [
            raw.map_elements(lambda value: filter_cnpj(value).cnpj, return_dtype=pl.Utf8).alias(f"{prefix}_cnpj"),
            raw.map_elements(lambda value: filter_cnpj(value).state, return_dtype=pl.Utf8).alias(f"{prefix}_cnpj_state"),
        ]

    admission_dt = parsed_date("DT_INTER", "DTINTERN", "admission_date")
    discharge_dt = parsed_date("DT_SAIDA", "DTSAIDA", "discharge_date")
    principal_raw = clean("DIAG_PRINC", "principal_icd")
    principal_norm = principal_raw.str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "").str.extract(r"([A-Z][0-9]{2}[0-9A-Z]?)", 1)
    death_raw = clean("MORTE", "OBITO")
    stay = nonnegative_int("stay_length_days", "DIAS_PERM", "QT_DIARIAS", "stay_length_days")
    icu_month = nonnegative_int("icu_days_month_total", "UTI_MES_TO")
    icu_adm = nonnegative_int("icu_days_hospitalization_total", "UTI_INT_TO")
    out = df.with_columns(
        admission_dt.alias("_admission_dt"),
        discharge_dt.alias("_discharge_dt"),
        admission_dt.dt.year().alias("admission_year"),
        *municipality("mun_residence", "MUNIC_RES", "CODMUNRES", "municipality"),
        *municipality("mun_movement", "MUNIC_MOV", "MUNIC_MOVI", "mun_movement"),
        principal_raw.alias("principal_icd_raw"),
        principal_norm.alias("principal_icd_norm"),
        *stay,
        *icu_month,
        *icu_adm,
        *cnpj_state("hospital", "CGC_HOSP", "hospital_cnpj"),
        *cnpj_state("maintainer", "CNPJ_MANT", "GESTOR_CPF", "maintainer_cnpj"),
    ).with_columns(
        pl.concat_str([pl.lit("SIH-"), pl.coalesce([clean("AIH", "N_AIH", "admission_id"), pl.col("_row_idx").cast(pl.Utf8) + pl.lit(f"_{source_manifest_hash[:8]}")])]).alias("admission_id"),
        pl.lit("SIH-RD").alias("source_system"),
        pl.col("_admission_dt").cast(pl.Utf8).alias("admission_date"),
        pl.col("_discharge_dt").cast(pl.Utf8).alias("discharge_date"),
        pl.when(clean("DT_INTER", "DTINTERN", "admission_date").is_null()).then(pl.lit("missing"))
        .when(pl.col("_admission_dt").is_null()).then(pl.lit("invalid"))
        .otherwise(pl.lit("valid")).alias("admission_date_state"),
        pl.when(clean("DT_SAIDA", "DTSAIDA", "discharge_date").is_null()).then(pl.lit("missing"))
        .when(pl.col("_discharge_dt").is_null()).then(pl.lit("invalid"))
        .otherwise(pl.lit("valid")).alias("discharge_date_state"),
        clean("CNES", "facility_code").alias("facility_cnes"),
        pl.lit(None, dtype=pl.Float64).alias("age_days"),
        clean("IDADE", "age").cast(pl.Float64, strict=False).alias("age_years"),
        clean("COD_IDADE", "CODIDADE").alias("age_unit"),
        pl.when(clean("IDADE", "age").is_null()).then(pl.lit("MissingAge")).otherwise(pl.lit("valid")).alias("age_state"),
        pl.when(digits("SEXO") == "1").then(pl.lit("male"))
        .when(digits("SEXO") == "2").then(pl.lit("female"))
        .otherwise(None).alias("sex"),
        pl.when(digits("SEXO").is_null()).then(pl.lit("missing"))
        .when(digits("SEXO").is_in(["1", "2"])).then(pl.lit("valid"))
        .when(digits("SEXO") == "9").then(pl.lit("unknown"))
        .otherwise(pl.lit("invalid")).alias("sex_state"),
        clean("RACA_COR").alias("race_color_billing"),
        pl.lit("sih_billing_race_color").alias("race_axis_type"),
        pl.when(pl.col("principal_icd_norm").is_null()).then(pl.lit("missing")).otherwise(pl.lit("valid")).alias("principal_icd_parse_state"),
        pl.lit("{}", dtype=pl.Utf8).alias("secondary_icd_raw_json"),
        pl.lit("{}", dtype=pl.Utf8).alias("secondary_icd_norm_json"),
        pl.lit("{}", dtype=pl.Utf8).alias("secondary_icd_parse_states_json"),
        pl.lit("{}", dtype=pl.Utf8).alias("secondary_diagnosis_type_json"),
        clean("PROC_SOLIC").alias("procedure_requested"),
        clean("PROC_REA").alias("procedure_performed"),
        clean("MARCA_UTI").alias("icu_type_mark"),
        pl.when(death_raw.str.to_lowercase().is_in(["1", "sim", "yes", "true", "t", "morte", "death"])).then(True)
        .when(death_raw.str.to_lowercase().is_in(["0", "não", "nao", "no", "false", "f"])).then(False)
        .otherwise(None).alias("death_flag"),
        pl.when(death_raw.is_null()).then(pl.lit("missing"))
        .when(death_raw.str.to_lowercase().is_in(["1", "sim", "yes", "true", "t", "morte", "death", "0", "não", "nao", "no", "false", "f"])).then(pl.lit("valid"))
        .otherwise(pl.lit("invalid")).alias("death_flag_state"),
        money("hospital_service_cost_real", "VAL_SH"),
        money("professional_service_cost_real", "VAL_SP"),
        money("icu_cost_real", "VAL_UTI"),
        money("total_admission_cost_real", "VAL_TOT"),
        pl.lit("{}", dtype=pl.Utf8).alias("cost_state_json"),
        pl.lit("{}", dtype=pl.Utf8).alias("cost_raw_json"),
        pl.when(pl.col("admission_year").is_not_null() & pl.col("mun_residence_cod6").is_not_null() & pl.col("principal_icd_norm").is_not_null())
        .then(pl.lit("valid")).otherwise(pl.lit("invalid_identity_or_principal_diagnosis")).alias("record_state"),
        pl.lit(source_manifest_hash).alias("source_manifest_hash"),
        pl.concat_str([pl.lit(source_manifest_hash), pl.lit(":"), pl.col("_row_idx").cast(pl.Utf8)]).hash().cast(pl.Utf8).alias("row_hash"),
        pl.lit(None, dtype=pl.Utf8).alias("raw_json"),
    ).rename({
        "hospital_cnpj": "hospital_cnpj",
        "maintainer_cnpj": "maintainer_cnpj",
        "mun_residence_state": "municipality_code_state",
    }).select([
        "admission_id", "source_system", "admission_date", "discharge_date", "admission_year",
        "admission_date_state", "discharge_date_state", "mun_residence_cod6", "mun_residence_cod7",
        "municipality_code_state", "mun_movement_cod6", "mun_movement_cod7", "mun_movement_state",
        "facility_cnes", "hospital_cnpj", "hospital_cnpj_state", "maintainer_cnpj", "maintainer_cnpj_state",
        "age_days", "age_years", "age_unit", "age_state", "sex", "sex_state", "race_color_billing",
        "race_axis_type", "principal_icd_raw", "principal_icd_norm", "principal_icd_parse_state",
        "secondary_icd_raw_json", "secondary_icd_norm_json", "secondary_icd_parse_states_json",
        "secondary_diagnosis_type_json", "procedure_requested", "procedure_performed", "stay_length_days",
        "stay_length_days_state", "icu_type_mark", "icu_days_month_total", "icu_days_month_total_state",
        "icu_days_hospitalization_total", "icu_days_hospitalization_total_state", "death_flag",
        "death_flag_state", "hospital_service_cost_real", "professional_service_cost_real", "icu_cost_real",
        "total_admission_cost_real", "cost_state_json", "cost_raw_json", "record_state",
        "source_manifest_hash", "row_hash", "raw_json",
    ]).rename({
        "stay_length_days_state": "stay_length_state",
        "icu_days_month_total_state": "icu_days_month_state",
        "icu_days_hospitalization_total_state": "icu_days_hospitalization_state",
    })
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return {
        "row_count": out.height,
        "output_path": str(out_path),
        "column_count": len(out.columns),
        "columns": out.columns,
        "deaths": int(out.get_column("death_flag").cast(pl.Int64, strict=False).fill_null(0).sum()) if out.height else 0,
        "cost_components": list(COST_COMPONENTS),
    }
