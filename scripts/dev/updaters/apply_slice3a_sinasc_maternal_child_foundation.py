from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path.cwd()

CREATE_OR_REPLACE = [
    "src/pegasus/datasus/sinasc_normalize.py",
    "src/pegasus/she/maternal_child.py",
    "src/pegasus/output/sinasc_efg_bundle.py",
    "src/pegasus/workflows/sinasc.py",
    "tests/fixtures/datasus/sinasc_fixture.csv",
    "tests/unit/test_sinasc_normalize.py",
    "tests/unit/test_sinasc_maternal_child_fields.py",
    "tests/unit/test_slice3a_cli_boundaries.py",
    "tests/integration/test_slice3a_sinasc_integration.py",
    "scripts/dev/audits/audit_slice3a_sinasc_maternal_child.py",
]

PATCH = ["src/pegasus/cli.py"]


def rel(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    return rel(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = rel(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def require_files(paths: list[str]) -> None:
    missing = [p for p in paths if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 3A preflight failed; missing expected files: {missing}")


def require_text(path: str, markers: list[str]) -> None:
    text = read(path)
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise RuntimeError(f"Slice 3A preflight failed; {path} missing markers: {missing}")


def require_python_symbol(path: str, symbol: str) -> None:
    source = read(path)
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    if symbol not in names:
        raise RuntimeError(f"Slice 3A preflight failed; {path} does not define {symbol}")


def preflight() -> None:
    require_files([
        "pyproject.toml",
        "src/pegasus/cli.py",
        "src/pegasus/datasus/normalize.py",
        "src/pegasus/datasus/icd_parser.py",
        "src/pegasus/output/bundle.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/workflows/compile.py",
        "tests/fixtures/datasus/sim_do_fixture.csv",
        "scripts/dev/audits/audit_slice2d_compile_smoke.py",
    ])
    require_text("src/pegasus/cli.py", [
        "datasus_app",
        "efg_app",
        "def datasus_normalize_sim",
        "def efg_build_sim_fixture",
        "def compile(",
    ])
    require_text("src/pegasus/output/validate.py", [
        "validate_output_bundle",
        "VariableDictionary",
        "Q_tensor",
        "ReproducibilityManifest",
    ])
    require_text("src/pegasus/output/reproducibility.py", [
        "COMPILE_TELEMETRY_STAGES",
        "RunTelemetry",
    ])
    require_python_symbol("src/pegasus/workflows/compile.py", "run_compile")


def patch_cli() -> None:
    path = rel("src/pegasus/cli.py")
    cli = path.read_text(encoding="utf-8")

    import_line = "from pegasus.workflows.acquire.sinasc import run_build_sinasc_fixture, run_datasus_normalize_sinasc\n"
    if import_line not in cli:
        marker = "from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture\n"
        if marker not in cli:
            raise RuntimeError("Cannot patch CLI imports: EFG workflow import marker not found.")
        cli = cli.replace(marker, marker + import_line, 1)

    normalize_cmd = '''

@datasus_app.command("normalize-sinasc")
def datasus_normalize_sinasc(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
) -> None:
    result = run_datasus_normalize_sinasc(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
    print(f"[green]sinasc normalized[/green] rows={result['row_count']} output={result['output_path']}")
'''
    if "def datasus_normalize_sinasc" not in cli:
        marker = '\n\n@efg_app.command("build-sim-fixture")'
        if marker not in cli:
            raise RuntimeError("Cannot patch CLI: build-sim-fixture command marker not found.")
        cli = cli.replace(marker, textwrap.dedent(normalize_cmd) + marker, 1)

    efg_cmd = '''

@efg_app.command("build-sinasc-fixture")
def efg_build_sinasc_fixture(
    sinasc_events: Path = typer.Option(..., "--sinasc-events"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    result = run_build_sinasc_fixture(
        sinasc_events_path=sinasc_events,
        run_dir=run_dir,
        municipality_cod6=municipality_cod6,
    )
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]SINASC maternal-child EFG bundle valid[/green] {result['run_dir']}")
'''
    if "def efg_build_sinasc_fixture" not in cli:
        marker = '\n\n@efg_app.command("attach-sidra-denominator")'
        if marker not in cli:
            raise RuntimeError("Cannot patch CLI: attach-sidra-denominator command marker not found.")
        cli = cli.replace(marker, textwrap.dedent(efg_cmd) + marker, 1)

    path.write_text(cli, encoding="utf-8", newline="\n")


def main() -> None:
    preflight()
    print("Slice 3A updater preflight passed.")
    print("CREATE/REPLACE:")
    for p in CREATE_OR_REPLACE:
        print(f"  {p}")
    print("PATCH:")
    for p in PATCH:
        print(f"  {p}")

    write("src/pegasus/datasus/sinasc_normalize.py", r'''
    from __future__ import annotations

    import hashlib
    import json
    import re
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    import polars as pl

    ICD_LIKE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]?")


    def _clean(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return text


    def _digits(value: Any) -> str | None:
        text = _clean(value)
        if text is None:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or None


    def _stable_hash(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


    def _read_table(path: str | Path) -> pl.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.read_parquet(path)
        if suffix == ".csv":
            return pl.read_csv(path, infer_schema_length=0, ignore_errors=False)
        raise ValueError(f"Unsupported SINASC input format: {path}")


    def _raw(row: dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in row:
                return row[name]
        return None


    def parse_sinasc_date(value: Any) -> tuple[str | None, int | None, str]:
        text = _clean(value)
        if text is None:
            return None, None, "missing"
        digits = _digits(text)
        candidates: list[str] = []
        if digits and len(digits) == 8:
            # SINASC DBF exports may appear either as YYYYMMDD-like or DDMMYYYY-like strings.
            candidates.append(f"{digits[0:4]}-{digits[4:6]}-{digits[6:8]}")
            candidates.append(f"{digits[4:8]}-{digits[2:4]}-{digits[0:2]}")
        candidates.append(text)
        for candidate in candidates:
            try:
                dt = datetime.fromisoformat(candidate).date()
                return dt.isoformat(), dt.year, "valid"
            except Exception:
                pass
        return None, None, "invalid"


    def municipality_codes(value: Any) -> tuple[str | None, str | None, str]:
        digits = _digits(value)
        if digits is None:
            return None, None, "missing"
        if len(digits) == 6:
            return digits, None, "datasus_cod6"
        if len(digits) == 7:
            return digits[:6], digits, "ibge_cod7"
        return None, None, "invalid"


    def int_or_none(value: Any) -> int | None:
        digits = _digits(value)
        if digits is None:
            return None
        try:
            return int(digits)
        except ValueError:
            return None


    def decode_count_preserve_leading_zero(
        value: Any,
        *,
        sentinels: set[str] | None = None,
        upper: int | None = None,
    ) -> tuple[int | None, str, str | None]:
        raw = _clean(value)
        if raw is None:
            return None, "missing", None
        sentinels = sentinels or set()
        digits = _digits(raw)
        if digits is None:
            return None, "invalid", raw
        if digits in sentinels:
            return None, "sentinel", digits
        value_int = int(digits)
        if upper is not None and value_int > upper:
            return None, "invalid", digits
        return value_int, "valid", digits


    def decode_birth_weight(value: Any) -> tuple[int | None, str, bool | None]:
        weight = int_or_none(value)
        if weight is None:
            return None, "missing", None
        if weight in {0, 9999}:
            return None, "sentinel", None
        if weight < 300 or weight > 7000:
            return None, "invalid", None
        return weight, "valid", weight < 2500


    def decode_gestational_age(value: Any) -> tuple[int | None, str, bool | None]:
        weeks = int_or_none(value)
        if weeks is None:
            return None, "missing", None
        if weeks in {0, 99}:
            return None, "sentinel", None
        if weeks < 20 or weeks > 45:
            return None, "invalid", None
        return weeks, "valid", weeks < 37


    def decode_apgar(value: Any) -> tuple[int | None, str, bool | None]:
        score = int_or_none(value)
        if score is None:
            return None, "missing", None
        if score == 99:
            return None, "sentinel", None
        if score < 0 or score > 10:
            return None, "invalid", None
        return score, "valid", score < 7


    def decode_delivery_mode(value: Any) -> tuple[str | None, str, bool | None]:
        code = _digits(value)
        if code is None:
            return None, "missing", None
        if code == "9":
            return code, "sentinel", None
        if code == "1":
            return code, "valid", False
        if code == "2":
            return code, "valid", True
        return code, "invalid", None


    def decode_race(value: Any) -> tuple[str | None, str]:
        code = _digits(value)
        if code is None:
            return None, "missing"
        if code in {"1", "2", "3", "4", "5"}:
            return code, "valid_admin_race"
        if code == "9":
            return None, "ignored_sentinel"
        return code, "invalid"


    def decode_anomaly_flag(value: Any) -> tuple[bool | None, str]:
        code = _digits(value)
        if code is None:
            return None, "missing"
        if code == "1":
            return False, "valid_absent"
        if code == "2":
            return True, "valid_present"
        if code == "9":
            return None, "sentinel"
        return None, "invalid"


    def normalize_anomaly_icd(value: Any, anomaly_flag: bool | None) -> tuple[str | None, str, bool]:
        text = _clean(value)
        if text is None:
            if anomaly_flag is True:
                return None, "flag_present_code_missing", True
            if anomaly_flag is False:
                return None, "absent", False
            return None, "unknown", False
        token = re.split(r"[;|,\s]+", text.upper().strip())[0]
        match = ICD_LIKE.match(token)
        if not match:
            return None, "invalid", bool(anomaly_flag)
        code = match.group(0)
        return code, "valid", code.startswith("Q") or bool(anomaly_flag)


    def normalize_sinasc_record(row: dict[str, Any], *, source_manifest_hash: str) -> dict[str, Any]:
        raw_payload = {str(k): v for k, v in row.items()}
        birth_date, birth_year, birth_date_state = parse_sinasc_date(_raw(row, "DTNASC", "DT_NASC", "NASCIMENTO"))
        mun_cod6, mun_cod7, mun_state = municipality_codes(_raw(row, "CODMUNRES", "CODMUNNASC", "MUN_RES"))
        weight, weight_state, low_weight = decode_birth_weight(_raw(row, "PESO", "PESO_NASC"))
        gest_weeks, gest_state, preterm = decode_gestational_age(_raw(row, "SEMAGESTAC", "GESTACAO", "QTSEMANAS"))
        apgar1, apgar1_state, low_apgar1 = decode_apgar(_raw(row, "APGAR1", "APGAR_1"))
        apgar5, apgar5_state, low_apgar5 = decode_apgar(_raw(row, "APGAR5", "APGAR_5"))
        delivery_code, delivery_state, cesarean = decode_delivery_mode(_raw(row, "PARTO", "TPPARTO"))
        consultations, consultations_state, consultations_raw = decode_count_preserve_leading_zero(
            _raw(row, "CONSULTAS", "QTCONSULTAS", "CONSPRENAT"),
            sentinels={"99"},
            upper=40,
        )
        mother_age = int_or_none(_raw(row, "IDADEMAE", "IDADE_MAE"))
        mother_race, mother_race_state = decode_race(_raw(row, "RACACORMAE", "RACA_COR_MAE"))
        newborn_race, newborn_race_state = decode_race(_raw(row, "RACACOR", "RACA_COR"))
        anomaly_flag, anomaly_flag_state = decode_anomaly_flag(_raw(row, "IDANOMAL", "ANOMALIA_FLAG"))
        anomaly_code, anomaly_state, anomaly_any = normalize_anomaly_icd(_raw(row, "CODANOMAL", "ANOMALIA"), anomaly_flag)

        event_key = _clean(_raw(row, "NUMERODN", "DN", "ID"))
        if event_key is None:
            event_key = _stable_hash(raw_payload)[:16]
        event_id = f"SINASC-{event_key}"

        record_state = "valid"
        if birth_date_state == "invalid" or mun_state == "invalid" or birth_year is None or mun_cod6 is None:
            record_state = "invalid_identity"

        return {
            "event_id": event_id,
            "birth_date": birth_date,
            "birth_year": birth_year,
            "birth_date_state": birth_date_state,
            "mun_residence_cod6": mun_cod6,
            "mun_residence_cod7": mun_cod7,
            "municipality_code_state": mun_state,
            "mother_age_years": mother_age,
            "adolescent_mother_flag": mother_age is not None and mother_age < 20,
            "advanced_maternal_age_flag": mother_age is not None and mother_age >= 35,
            "mother_race_admin_code": mother_race,
            "mother_race_state": mother_race_state,
            "newborn_race_admin_code": newborn_race,
            "newborn_race_state": newborn_race_state,
            "sex_code": _digits(_raw(row, "SEXO")),
            "birth_weight_g": weight,
            "birth_weight_state": weight_state,
            "low_birth_weight_flag": low_weight,
            "gestational_age_weeks": gest_weeks,
            "gestational_age_state": gest_state,
            "prematurity_flag": preterm,
            "apgar1": apgar1,
            "apgar1_state": apgar1_state,
            "low_apgar1_flag": low_apgar1,
            "apgar5": apgar5,
            "apgar5_state": apgar5_state,
            "low_apgar5_flag": low_apgar5,
            "delivery_mode_code": delivery_code,
            "delivery_mode_state": delivery_state,
            "cesarean_flag": cesarean,
            "prenatal_consult_count": consultations,
            "prenatal_consult_state": consultations_state,
            "prenatal_consult_raw_digits": consultations_raw,
            "insufficient_prenatal_flag": consultations is not None and consultations < 7,
            "anomaly_flag": anomaly_flag,
            "anomaly_flag_state": anomaly_flag_state,
            "anomaly_icd_raw": _clean(_raw(row, "CODANOMAL", "ANOMALIA")),
            "anomaly_icd_code": anomaly_code,
            "anomaly_icd_state": anomaly_state,
            "congenital_anomaly_flag": anomaly_any,
            "record_state": record_state,
            "source_manifest_hash": source_manifest_hash,
            "row_hash": _stable_hash(raw_payload),
            "raw_json": json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, default=str),
        }


    def normalize_sinasc_events(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, Any]:
        df = _read_table(input_path)
        rows = [normalize_sinasc_record(row, source_manifest_hash=source_manifest_hash) for row in df.to_dicts()]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(out)
        return {
            "row_count": len(rows),
            "output_path": str(out),
            "valid_rows": sum(1 for row in rows if row["record_state"] == "valid"),
            "low_birth_weight_rows": sum(1 for row in rows if row["low_birth_weight_flag"] is True),
            "prematurity_rows": sum(1 for row in rows if row["prematurity_flag"] is True),
            "cesarean_rows": sum(1 for row in rows if row["cesarean_flag"] is True),
            "low_apgar5_rows": sum(1 for row in rows if row["low_apgar5_flag"] is True),
            "insufficient_prenatal_rows": sum(1 for row in rows if row["insufficient_prenatal_flag"] is True),
            "anomaly_rows": sum(1 for row in rows if row["congenital_anomaly_flag"] is True),
        }
    ''')

    write("src/pegasus/she/maternal_child.py", r'''
    from __future__ import annotations

    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import polars as pl


    @dataclass(frozen=True)
    class MaternalChildSummary:
        years: list[int]
        municipalities_cod6: list[str]
        births_total: int
        low_birth_weight_births: int
        prematurity_births: int
        cesarean_births: int
        congenital_anomaly_births: int
        low_apgar5_births: int
        adolescent_mother_births: int
        advanced_maternal_age_births: int
        insufficient_prenatal_births: int
        birth_weight_missing_or_invalid: int
        gestational_age_missing_or_invalid: int
        apgar_missing_or_invalid: int
        race_missing_or_ignored: int

        def rates(self) -> dict[str, float | None]:
            denom = float(self.births_total) if self.births_total else 0.0
            names = {
                "low_birth_weight_prevalence": self.low_birth_weight_births,
                "prematurity_prevalence": self.prematurity_births,
                "cesarean_prevalence": self.cesarean_births,
                "congenital_anomaly_prevalence": self.congenital_anomaly_births,
                "low_apgar5_prevalence": self.low_apgar5_births,
                "adolescent_mother_share": self.adolescent_mother_births,
                "advanced_maternal_age_share": self.advanced_maternal_age_births,
                "insufficient_prenatal_share": self.insufficient_prenatal_births,
            }
            if denom <= 0:
                return {key: None for key in names}
            return {key: value / denom for key, value in names.items()}

        def support(self) -> dict[str, Any]:
            return {
                "time": {"years": self.years},
                "geography": {"municipality_cod6": self.municipalities_cod6},
                "n_events": self.births_total,
            }


    def _bool_count(df: pl.DataFrame, column: str) -> int:
        if column not in df.columns:
            return 0
        return int(df.filter(pl.col(column) == True).height)  # noqa: E712 - explicit data-state comparison


    def _state_not_valid_count(df: pl.DataFrame, column: str, valid_state: str = "valid") -> int:
        if column not in df.columns:
            return 0
        return int(df.filter(pl.col(column) != valid_state).height)


    def summarize_maternal_child_events(events_path: str | Path, *, municipality_cod6: str | None = None) -> MaternalChildSummary:
        df = pl.read_parquet(events_path)
        if municipality_cod6 is not None:
            df = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
        valid = df.filter(pl.col("record_state") == "valid") if "record_state" in df.columns else df

        years = sorted(int(x) for x in valid["birth_year"].drop_nulls().unique().to_list()) if "birth_year" in valid.columns else []
        municipalities = (
            sorted(str(x) for x in valid["mun_residence_cod6"].drop_nulls().unique().to_list())
            if "mun_residence_cod6" in valid.columns
            else []
        )
        race_missing = 0
        for col in ["mother_race_state", "newborn_race_state"]:
            if col in valid.columns:
                race_missing += int(valid.filter(pl.col(col) != "valid_admin_race").height)

        return MaternalChildSummary(
            years=years,
            municipalities_cod6=municipalities,
            births_total=int(valid.height),
            low_birth_weight_births=_bool_count(valid, "low_birth_weight_flag"),
            prematurity_births=_bool_count(valid, "prematurity_flag"),
            cesarean_births=_bool_count(valid, "cesarean_flag"),
            congenital_anomaly_births=_bool_count(valid, "congenital_anomaly_flag"),
            low_apgar5_births=_bool_count(valid, "low_apgar5_flag"),
            adolescent_mother_births=_bool_count(valid, "adolescent_mother_flag"),
            advanced_maternal_age_births=_bool_count(valid, "advanced_maternal_age_flag"),
            insufficient_prenatal_births=_bool_count(valid, "insufficient_prenatal_flag"),
            birth_weight_missing_or_invalid=_state_not_valid_count(valid, "birth_weight_state"),
            gestational_age_missing_or_invalid=_state_not_valid_count(valid, "gestational_age_state"),
            apgar_missing_or_invalid=_state_not_valid_count(valid, "apgar5_state"),
            race_missing_or_ignored=race_missing,
        )
    ''')

    write("src/pegasus/output/sinasc_efg_bundle.py", r'''
    from __future__ import annotations

    import json
    import platform
    import sys
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import pyarrow as pa
    import pyarrow.parquet as pq
    import polars as pl

    from pegasus.core.hashing import sha256_file, sha256_text
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
    from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
    from pegasus.she.maternal_child import MaternalChildSummary, summarize_maternal_child_events


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


    def _empty_like(path: Path) -> None:
        table = pq.read_table(path)
        pq.write_table(table.slice(0, 0), path)


    def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
        schema = pq.read_table(path).schema
        fixed = [{field.name: row.get(field.name) for field in schema} for row in rows]
        table = pa.Table.from_pylist(fixed, schema=schema) if fixed else pa.Table.from_pylist([], schema=schema)
        pq.write_table(table, path)


    def _field_row(
        *,
        field_id: str,
        name: str,
        kind: str,
        unit: str,
        aggregation: str,
        role: list[str],
        support: dict[str, Any],
        axes: dict[str, Any],
        operator: str | None,
        provenance: list[str],
        warnings: list[str],
        state: str = "verified",
        path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "field_id": field_id,
            "name": name,
            "kind": kind,
            "carrier": "LiveBirths",
            "unit": unit,
            "aggregation": aggregation,
            "role": _json(role),
            "source": _json(["SINASC"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": operator,
            "provenance": _json(provenance),
            "state": state,
            "dashboard_safe": "True" if state == "verified" else "False",
            "warnings": _json(warnings),
            "lineage_hash": sha256_text(_json({"field_id": field_id, "operator": operator, "support": support})),
            "registry_hash": "sinasc_maternal_child_registry_v1",
            "materialization_state": "materialized",
            "path": path,
        }


    def _q_row(*, field: dict[str, Any], n_events: int | float | None, n_denom: int | float | None, warnings: list[str]) -> dict[str, Any]:
        n_eff = n_denom if n_denom is not None else n_events
        missingness = 0.0
        if "decoder_missingness" in field:
            missingness = float(field["decoder_missingness"])
        return {
            "field_id": field["field_id"],
            "n_events": float(n_events) if n_events is not None else None,
            "n_denom": float(n_denom) if n_denom is not None else None,
            "n_eff": float(n_eff) if n_eff is not None else None,
            "cov_S": None,
            "cov_T": None,
            "missingness": missingness,
            "zero_inflation": 0.0,
            "denom_fragility": 0.0 if n_denom else None,
            "provenance_risk": 0.15,
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "warnings": _json(warnings),
            "computed_at": _now(),
            "q_schema_version": "v1",
        }


    def _vd_row(*, field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
        return {
            "field_id": field["field_id"],
            "display_name": field["name"],
            "technical_name": field["field_id"],
            "definition": definition,
            "estimand_label": estimand,
            "source_systems": _json(["SINASC"]),
            "carrier": field["carrier"],
            "unit": field["unit"],
            "support_description": "SINASC live-birth event support by birth year and municipality of residence.",
            "axis_description": "Maternal and newborn administrative race axes are preserved separately; no race bridge or IBGE denominator replacement is applied.",
            "provenance_description": "Derived from normalized SINASC DN records with explicit decoder states.",
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "interpretation_warning": warning,
        }


    def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "edge_id": edge_id,
            "parent_field_id": parent,
            "child_field_id": child,
            "operator": operator,
            "operator_params_json": _json(params),
            "registry_versions_json": _json({"maternal_child": "v1"}),
            "created_at": _now(),
        }


    def _failed_branch_rows(summary: MaternalChildSummary) -> list[dict[str, Any]]:
        parent_ids = _json(["sinasc_births_all"])
        return [
            {
                "branch_id": "sinasc_crude_birth_rate_requires_population_denominator",
                "parent_field_ids": parent_ids,
                "operator": "birth_rate",
                "reason": "blocked_missing_population_denominator_anchor",
                "severity": "blocked",
                "created_at": _now(),
                "details_json": _json({"births_total": summary.births_total, "required_source": "SIDRA population denominator"}),
            },
            {
                "branch_id": "sinasc_infant_mortality_requires_sim_linkage",
                "parent_field_ids": parent_ids,
                "operator": "infant_mortality_rate",
                "reason": "blocked_missing_sim_death_numerator_linkage",
                "severity": "blocked",
                "created_at": _now(),
                "details_json": _json({"births_total": summary.births_total, "required_source": "SIM-DO"}),
            },
            {
                "branch_id": "sinasc_neonatal_mortality_requires_sim_linkage",
                "parent_field_ids": parent_ids,
                "operator": "neonatal_mortality_rate",
                "reason": "blocked_missing_sim_neonatal_death_numerator_linkage",
                "severity": "blocked",
                "created_at": _now(),
                "details_json": _json({"births_total": summary.births_total, "required_source": "SIM-DO"}),
            },
            {
                "branch_id": "sinasc_postneonatal_mortality_requires_sim_linkage",
                "parent_field_ids": parent_ids,
                "operator": "postneonatal_mortality_rate",
                "reason": "blocked_missing_sim_postneonatal_death_numerator_linkage",
                "severity": "blocked",
                "created_at": _now(),
                "details_json": _json({"births_total": summary.births_total, "required_source": "SIM-DO"}),
            },
        ]


    def _warning_rows(summary: MaternalChildSummary) -> list[dict[str, Any]]:
        return [
            {
                "warning_id": "sinasc_race_axes_preserved_separately",
                "field_id": "run",
                "severity": "info",
                "message": "Maternal and newborn administrative race axes remain separate; no IBGE self-declared denominator replacement or Bridge_R has been applied.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "sinasc_mortality_fields_blocked_without_sim_linkage",
                "field_id": "run",
                "severity": "blocked",
                "message": "Infant, neonatal, and postneonatal mortality require SIM numerator linkage and are emitted as FailedBranches in this slice.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "sinasc_birth_rate_blocked_without_population_denominator",
                "field_id": "run",
                "severity": "blocked",
                "message": "Crude birth rate requires a legal population denominator anchor and is emitted as a FailedBranch in this foundation slice.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "sinasc_decoder_state_counts",
                "field_id": "run",
                "severity": "info",
                "message": _json({
                    "birth_weight_missing_or_invalid": summary.birth_weight_missing_or_invalid,
                    "gestational_age_missing_or_invalid": summary.gestational_age_missing_or_invalid,
                    "apgar_missing_or_invalid": summary.apgar_missing_or_invalid,
                    "race_missing_or_ignored": summary.race_missing_or_ignored,
                }),
                "created_at": _now(),
                "inherited_from": None,
            },
        ]


    def _count_specs(summary: MaternalChildSummary) -> list[tuple[str, str, int, str, str]]:
        return [
            ("sinasc_births_all", "SINASC Live Births", summary.births_total, "Live-birth count from normalized SINASC records.", "live_birth_count"),
            ("sinasc_low_birth_weight_births", "Low Birth Weight Births", summary.low_birth_weight_births, "Births with valid birth weight below 2500 g.", "low_birth_weight_count"),
            ("sinasc_prematurity_births", "Prematurity Births", summary.prematurity_births, "Births with valid gestational age below 37 completed weeks.", "prematurity_count"),
            ("sinasc_cesarean_births", "Cesarean Births", summary.cesarean_births, "Births with SINASC delivery mode decoded as cesarean.", "cesarean_count"),
            ("sinasc_congenital_anomaly_births", "Congenital Anomaly Births", summary.congenital_anomaly_births, "Births with a valid congenital anomaly flag or Q-prefix anomaly code.", "congenital_anomaly_count"),
            ("sinasc_low_apgar5_births", "Low 5-Minute APGAR Births", summary.low_apgar5_births, "Births with valid 5-minute APGAR below 7.", "low_apgar5_count"),
            ("sinasc_adolescent_mother_births", "Adolescent Mother Births", summary.adolescent_mother_births, "Births where maternal age is valid and below 20 years.", "adolescent_mother_count"),
            ("sinasc_advanced_maternal_age_births", "Advanced Maternal Age Births", summary.advanced_maternal_age_births, "Births where maternal age is valid and at least 35 years.", "advanced_maternal_age_count"),
            ("sinasc_insufficient_prenatal_births", "Insufficient Prenatal Consultation Births", summary.insufficient_prenatal_births, "Births with a valid prenatal consultation count below 7.", "insufficient_prenatal_count"),
        ]


    def _rate_specs(summary: MaternalChildSummary) -> list[tuple[str, str, int, str, str, str]]:
        return [
            ("sinasc_low_birth_weight_prevalence", "Low Birth Weight Prevalence", summary.low_birth_weight_births, "sinasc_low_birth_weight_births", "Low birth weight births divided by all live births.", "low_birth_weight_prevalence"),
            ("sinasc_prematurity_prevalence", "Prematurity Prevalence", summary.prematurity_births, "sinasc_prematurity_births", "Prematurity births divided by all live births.", "prematurity_prevalence"),
            ("sinasc_cesarean_prevalence", "Cesarean Birth Prevalence", summary.cesarean_births, "sinasc_cesarean_births", "Cesarean births divided by all live births.", "cesarean_prevalence"),
            ("sinasc_congenital_anomaly_prevalence", "Congenital Anomaly Prevalence", summary.congenital_anomaly_births, "sinasc_congenital_anomaly_births", "Congenital anomaly births divided by all live births.", "congenital_anomaly_prevalence"),
            ("sinasc_low_apgar5_prevalence", "Low 5-Minute APGAR Prevalence", summary.low_apgar5_births, "sinasc_low_apgar5_births", "Low 5-minute APGAR births divided by all live births.", "low_apgar5_prevalence"),
            ("sinasc_adolescent_mother_share", "Adolescent Mother Share", summary.adolescent_mother_births, "sinasc_adolescent_mother_births", "Births to mothers under 20 divided by all live births.", "adolescent_mother_share"),
            ("sinasc_advanced_maternal_age_share", "Advanced Maternal Age Share", summary.advanced_maternal_age_births, "sinasc_advanced_maternal_age_births", "Births to mothers at least 35 divided by all live births.", "advanced_maternal_age_share"),
            ("sinasc_insufficient_prenatal_share", "Insufficient Prenatal Consultation Share", summary.insufficient_prenatal_births, "sinasc_insufficient_prenatal_births", "Births with fewer than 7 prenatal consultations divided by all live births.", "insufficient_prenatal_share"),
        ]


    def build_sinasc_fixture_rows(summary: MaternalChildSummary, *, events_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        support = summary.support()
        axes = {
            "time_axis": "birth_year",
            "geo_axis": "mun_residence_cod6",
            "event_axis": "SINASC_DN",
            "maternal_race_axis": "RACACORMAE_admin",
            "newborn_race_axis": "RACACOR_admin",
            "anomaly_axis": "IDANOMAL_plus_CODANOMAL",
        }
        path = "Tables/sinasc_maternal_child_summary.parquet"
        shared_warning = ["sinasc_fixture_support", "race_axes_not_bridged", "mortality_linkage_blocked"]
        fields: list[dict[str, Any]] = []
        q_rows: list[dict[str, Any]] = []
        vd_rows: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for field_id, name, value, definition, estimand in _count_specs(summary):
            field = _field_row(
                field_id=field_id,
                name=name,
                kind="extensive_measure",
                unit="births",
                aggregation="additive_count",
                role=["exposure", "outcome_candidate", "maternal_child"],
                support={**support, "n_events": value},
                axes=axes,
                operator="sinasc_event_count" if field_id != "sinasc_births_all" else "identity_count",
                provenance=["SINASC", str(events_path)],
                warnings=shared_warning,
                path=path,
            )
            fields.append(field)
            q_rows.append(_q_row(field=field, n_events=value, n_denom=None, warnings=shared_warning))
            vd_rows.append(_vd_row(field=field, definition=definition, estimand=estimand, warning="Fixture-level maternal-child field; mortality outputs require SIM linkage."))

        for field_id, name, numerator_value, numerator_id, definition, estimand in _rate_specs(summary):
            field = _field_row(
                field_id=field_id,
                name=name,
                kind="intensive_density",
                unit="proportion",
                aggregation="ratio_recomputed_from_counts",
                role=["outcome_candidate", "descriptive_rate", "maternal_child"],
                support={**support, "n_events": numerator_value, "n_denom": summary.births_total},
                axes={**axes, "denominator_field_id": "sinasc_births_all", "numerator_field_id": numerator_id},
                operator="ratio_from_aligned_sinasc_counts",
                provenance=["SINASC", str(events_path)],
                warnings=shared_warning,
                path=path,
            )
            fields.append(field)
            q_rows.append(_q_row(field=field, n_events=numerator_value, n_denom=summary.births_total, warnings=shared_warning))
            vd_rows.append(_vd_row(field=field, definition=definition, estimand=estimand, warning="Ratio is recomputed from aligned SINASC count numerator and live-birth denominator."))
            edges.append(_edge(f"edge_{numerator_id}_to_{field_id}", numerator_id, field_id, "ratio_numerator", {"law": "count_over_births"}))
            edges.append(_edge(f"edge_sinasc_births_all_to_{field_id}", "sinasc_births_all", field_id, "ratio_denominator", {"law": "count_over_births"}))

        return fields, q_rows, vd_rows, edges


    def _summary_table(summary: MaternalChildSummary) -> list[dict[str, Any]]:
        return [{
            "births_total": summary.births_total,
            "low_birth_weight_births": summary.low_birth_weight_births,
            "prematurity_births": summary.prematurity_births,
            "cesarean_births": summary.cesarean_births,
            "congenital_anomaly_births": summary.congenital_anomaly_births,
            "low_apgar5_births": summary.low_apgar5_births,
            "adolescent_mother_births": summary.adolescent_mother_births,
            "advanced_maternal_age_births": summary.advanced_maternal_age_births,
            "insufficient_prenatal_births": summary.insufficient_prenatal_births,
            **summary.rates(),
        }]


    def write_sinasc_fixture_efg_bundle(*, sinasc_events_path: str | Path, run_dir: str | Path, municipality_cod6: str | None = None) -> Path:
        events_path = Path(sinasc_events_path)
        run_dir = create_empty_output_bundle(run_dir)
        summary = summarize_maternal_child_events(events_path, municipality_cod6=municipality_cod6)
        fields, q_rows, vd_rows, edges = build_sinasc_fixture_rows(summary, events_path=events_path)

        pl.DataFrame(_summary_table(summary)).write_parquet(run_dir / "Tables" / "sinasc_maternal_child_summary.parquet")

        _write_rows_like(run_dir / "V_fields.parquet", fields)
        _write_rows_like(run_dir / "Q_tensor.parquet", q_rows)
        _write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows)
        _write_rows_like(run_dir / "E_DAG.parquet", edges)
        _write_rows_like(run_dir / "Warnings.parquet", _warning_rows(summary))
        _write_rows_like(run_dir / "FailedBranches.parquet", _failed_branch_rows(summary))

        for name in ["ModelAssociations.parquet", "ResidualAssociations.parquet", "Hypotheses.parquet", "QuarantinedFields.parquet", "ForcedFields.parquet"]:
            _empty_like(run_dir / name)

        user_intent = {
            "geography": {"level": "municipality", "codes": summary.municipalities_cod6, "uf": ["AL"]},
            "time": {"start_year": min(summary.years) if summary.years else 2022, "end_year": max(summary.years) if summary.years else 2022},
            "health_seeds": ["maternal_child"],
            "mandatory_fields": [row["field_id"] for row in fields],
            "system_weights": {"SINASC": 1.0},
            "context_policy": [],
            "budget": "fast",
            "geo_mode": "native",
            "force_selectors": [],
            "exclude_systems": [],
            "execution_scale": "smoke",
            "race_tensor_mode": "decoupled",
            "population_mode": "blocked_missing",
        }
        run_config = {
            "run_id": Path(run_dir).name,
            "workflow": "sinasc_fixture_maternal_child_efg",
            "source_hashes": {"sinasc_events": sha256_file(events_path)},
            "registry_hashes": {"maternal_child": "sinasc_maternal_child_registry_v1"},
            "municipality_filter_cod6": municipality_cod6,
            "created_at": _now(),
        }
        stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
        stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
        for stage in ["datasus_normalize", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
            if stage in stage_status:
                stage_status[stage] = "success"
        for stage in ["population_solver", "stdfm", "pirs_model", "pirs_hsic"]:
            if stage in stage_status:
                stage_status[stage] = "blocked"
        telemetry = {
            "total_wall_seconds": 0.0,
            "stage_status": stage_status,
            "stage_wall_seconds": stage_wall_seconds,
            "stage_errors": {
                "population_solver": "No population denominator tensor invoked in Slice 3A SINASC fixture path.",
                "stdfm": "ST-DFM scaffold remains blocked for Slice 3A SINASC fixture path.",
                "pirs_model": "PIRS model fitting is outside Slice 3A.",
                "pirs_hsic": "HSIC scanning is outside Slice 3A.",
            },
            "resource_summary": {
                "peak_rss_mb": None,
                "peak_vram_mb": None,
                "duckdb_temp_bytes": None,
                "rows_read": {"sinasc_events": summary.births_total},
                "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows), "FailedBranches": 4},
                "parquet_bytes_written": 0,
            },
        }
        manifest = {
            "run_id": Path(run_dir).name,
            "created_at": _now(),
            "workflow": "sinasc_fixture_maternal_child_efg",
            "source_hashes": run_config["source_hashes"],
            "registry_hashes": run_config["registry_hashes"],
            "telemetry": telemetry,
            "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
            "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows), "FailedBranches": 4},
        }
        p_vector = {
            "source_systems": ["SINASC"],
            "source_hashes": run_config["source_hashes"],
            "registry_hashes": run_config["registry_hashes"],
            "field_count": len(fields),
            "blocked_outputs": ["crude_birth_rate", "infant_mortality", "neonatal_mortality", "postneonatal_mortality"],
            "provenance": {row["field_id"]: ["SINASC", "normalized_dn_fixture", "maternal_child_field"] for row in fields},
        }
        (run_dir / "UserIntent.json").write_text(_json(user_intent), encoding="utf-8")
        (run_dir / "RunConfig.json").write_text(_json(run_config), encoding="utf-8")
        (run_dir / "ReproducibilityManifest.json").write_text(_json(manifest), encoding="utf-8")
        (run_dir / "P_vector.json").write_text(_json(p_vector), encoding="utf-8")

        expected = set(OUTPUT_BUNDLE_FILES.values())
        found = {p.name for p in run_dir.iterdir()}
        extra = found - expected
        missing = expected - found
        if extra or missing:
            raise RuntimeError(f"Invalid first-class bundle keys. extra={sorted(extra)} missing={sorted(missing)}")
        return run_dir
    ''')

    write("src/pegasus/workflows/sinasc.py", r'''
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    from pegasus.datasus.sinasc_normalize import normalize_sinasc_events
    from pegasus.output.sinasc_efg_bundle import write_sinasc_fixture_efg_bundle
    from pegasus.output.validate import validate_output_bundle


    def run_datasus_normalize_sinasc(
        *,
        input_path: str | Path,
        output_path: str | Path,
        source_manifest_hash: str,
    ) -> dict[str, Any]:
        return normalize_sinasc_events(
            input_path=input_path,
            output_path=output_path,
            source_manifest_hash=source_manifest_hash,
        )


    def run_build_sinasc_fixture(
        *,
        sinasc_events_path: str | Path,
        run_dir: str | Path,
        municipality_cod6: str | None = None,
    ) -> dict[str, Any]:
        output = write_sinasc_fixture_efg_bundle(
            sinasc_events_path=sinasc_events_path,
            run_dir=run_dir,
            municipality_cod6=municipality_cod6,
        )
        validation = validate_output_bundle(run_dir=str(output))
        return {"run_dir": output, "validation": validation}
    ''')

    write("tests/fixtures/datasus/sinasc_fixture.csv", '''
    NUMERODN,CODMUNRES,DTNASC,IDADEMAE,RACACORMAE,RACACOR,SEXO,PESO,SEMAGESTAC,APGAR1,APGAR5,PARTO,CONSULTAS,IDANOMAL,CODANOMAL
    DN0001,270430,20220115,24,4,4,1,3200,39,8,9,2,07,1,
    DN0002,270430,20220210,17,4,4,2,2400,36,7,8,1,04,2,Q359
    DN0003,270030,20220305,30,1,1,1,1800,32,6,6,2,02,2,Q540
    DN0004,270030,20220407,29,9,,2,,99,99,99,9,99,9,
    DN0005,270430,20220511,36,4,4,2,2800,38,9,10,1,08,1,
    ''')

    write("tests/unit/test_sinasc_normalize.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.datasus.sinasc_normalize import normalize_sinasc_events


    def test_sinasc_fixture_normalization_preserves_decoder_states(tmp_path: Path):
        out = tmp_path / "sinasc_events.parquet"
        result = normalize_sinasc_events(
            input_path="tests/fixtures/datasus/sinasc_fixture.csv",
            output_path=out,
            source_manifest_hash="fixture_manifest",
        )
        assert result["row_count"] == 5
        assert result["valid_rows"] == 5
        assert result["low_birth_weight_rows"] == 2
        assert result["prematurity_rows"] == 2
        assert result["cesarean_rows"] == 2
        assert result["low_apgar5_rows"] == 1
        assert result["insufficient_prenatal_rows"] == 2
        assert result["anomaly_rows"] == 2

        df = pl.read_parquet(out)
        assert "raw_json" in df.columns
        assert df.filter(pl.col("event_id") == "SINASC-DN0002")["birth_weight_g"].item() == 2400
        assert df.filter(pl.col("event_id") == "SINASC-DN0002")["low_birth_weight_flag"].item() is True
        assert df.filter(pl.col("event_id") == "SINASC-DN0004")["birth_weight_state"].item() == "missing"
        assert df.filter(pl.col("event_id") == "SINASC-DN0004")["gestational_age_state"].item() == "sentinel"
        assert df.filter(pl.col("event_id") == "SINASC-DN0001")["prenatal_consult_count"].item() == 7
        assert df.filter(pl.col("event_id") == "SINASC-DN0001")["prenatal_consult_raw_digits"].item() == "07"
        assert df.filter(pl.col("event_id") == "SINASC-DN0003")["low_apgar5_flag"].item() is True
        assert df.filter(pl.col("event_id") == "SINASC-DN0005")["advanced_maternal_age_flag"].item() is True
    ''')

    write("tests/unit/test_sinasc_maternal_child_fields.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.datasus.sinasc_normalize import normalize_sinasc_events
    from pegasus.output.sinasc_efg_bundle import write_sinasc_fixture_efg_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.she.maternal_child import summarize_maternal_child_events


    def _events(tmp_path: Path) -> Path:
        out = tmp_path / "sinasc_events.parquet"
        normalize_sinasc_events(
            input_path="tests/fixtures/datasus/sinasc_fixture.csv",
            output_path=out,
            source_manifest_hash="fixture_manifest",
        )
        return out


    def test_maternal_child_summary_counts_fixture_events(tmp_path: Path):
        summary = summarize_maternal_child_events(_events(tmp_path))
        assert summary.births_total == 5
        assert summary.low_birth_weight_births == 2
        assert summary.prematurity_births == 2
        assert summary.cesarean_births == 2
        assert summary.congenital_anomaly_births == 2
        assert summary.low_apgar5_births == 1
        assert summary.adolescent_mother_births == 1
        assert summary.advanced_maternal_age_births == 1
        assert summary.insufficient_prenatal_births == 2
        assert summary.years == [2022]
        assert summary.municipalities_cod6 == ["270030", "270430"]


    def test_sinasc_efg_bundle_validates_and_blocks_missing_denominators_and_mortality(tmp_path: Path):
        events = _events(tmp_path)
        run_dir = write_sinasc_fixture_efg_bundle(sinasc_events_path=events, run_dir=tmp_path / "run")
        validation = validate_output_bundle(run_dir=str(run_dir))
        assert validation.ok, validation.errors

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        names = set(v["field_id"].to_list())
        assert "sinasc_births_all" in names
        assert "sinasc_low_birth_weight_prevalence" in names
        assert "sinasc_congenital_anomaly_prevalence" in names
        assert "sinasc_low_apgar5_prevalence" in names
        assert "sinasc_insufficient_prenatal_share" in names

        q = pl.read_parquet(run_dir / "Q_tensor.parquet")
        assert set(v["field_id"].to_list()).issubset(set(q["field_id"].to_list()))

        failed = pl.read_parquet(run_dir / "FailedBranches.parquet")
        reasons = set(failed["reason"].to_list())
        assert "blocked_missing_population_denominator_anchor" in reasons
        assert "blocked_missing_sim_death_numerator_linkage" in reasons
        assert "blocked_missing_sim_neonatal_death_numerator_linkage" in reasons
        assert "blocked_missing_sim_postneonatal_death_numerator_linkage" in reasons
    ''')

    write("tests/unit/test_slice3a_cli_boundaries.py", r'''
    from pathlib import Path

    from typer.testing import CliRunner

    from pegasus.cli import app


    def test_slice3a_cli_commands_delegate_to_workflows():
        cli = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
        assert "run_datasus_normalize_sinasc" in cli
        assert "run_build_sinasc_fixture" in cli
        assert '@datasus_app.command("normalize-sinasc")' in cli
        assert '@efg_app.command("build-sinasc-fixture")' in cli
        assert "normalize_sinasc_events(" not in cli
        assert "write_sinasc_fixture_efg_bundle(" not in cli


    def test_slice3a_cli_normalize_and_build_commands(tmp_path: Path):
        runner = CliRunner()
        events = tmp_path / "sinasc_events.parquet"
        result = runner.invoke(
            app,
            [
                "datasus", "normalize-sinasc",
                "--input", "tests/fixtures/datasus/sinasc_fixture.csv",
                "--output", str(events),
                "--source-manifest-hash", "fixture_manifest",
            ],
        )
        assert result.exit_code == 0, result.output
        assert events.exists()

        run_dir = tmp_path / "run"
        result = runner.invoke(
            app,
            [
                "efg", "build-sinasc-fixture",
                "--sinasc-events", str(events),
                "--run-dir", str(run_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (run_dir / "V_fields.parquet").exists()
    ''')

    write("tests/integration/test_slice3a_sinasc_integration.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.acquire.sinasc import run_build_sinasc_fixture, run_datasus_normalize_sinasc


    def test_slice3a_sinasc_normalize_to_maternal_child_bundle(tmp_path: Path):
        events = tmp_path / "processed" / "sinasc_events.parquet"
        normalize = run_datasus_normalize_sinasc(
            input_path="tests/fixtures/datasus/sinasc_fixture.csv",
            output_path=events,
            source_manifest_hash="fixture_manifest",
        )
        assert normalize["row_count"] == 5

        result = run_build_sinasc_fixture(
            sinasc_events_path=events,
            run_dir=tmp_path / "run",
        )
        assert result["validation"].ok, result["validation"].errors
        assert validate_output_bundle(run_dir=str(result["run_dir"])).ok

        table = pl.read_parquet(result["run_dir"] / "Tables" / "sinasc_maternal_child_summary.parquet")
        assert table["births_total"].item() == 5
        assert table["low_birth_weight_births"].item() == 2
        assert table["congenital_anomaly_births"].item() == 2
        assert table["insufficient_prenatal_births"].item() == 2


    def test_slice3a_sinasc_municipality_filter_changes_support(tmp_path: Path):
        events = tmp_path / "processed" / "sinasc_events.parquet"
        run_datasus_normalize_sinasc(
            input_path="tests/fixtures/datasus/sinasc_fixture.csv",
            output_path=events,
            source_manifest_hash="fixture_manifest",
        )
        result = run_build_sinasc_fixture(
            sinasc_events_path=events,
            run_dir=tmp_path / "run_maceio",
            municipality_cod6="270430",
        )
        assert result["validation"].ok, result["validation"].errors
        table = pl.read_parquet(result["run_dir"] / "Tables" / "sinasc_maternal_child_summary.parquet")
        assert table["births_total"].item() == 3
        assert table["low_birth_weight_births"].item() == 1
        assert table["congenital_anomaly_births"].item() == 1
    ''')

    write("scripts/dev/audits/audit_slice3a_sinasc_maternal_child.py", r'''
    from __future__ import annotations

    import argparse
    import json
    import sys
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle


    def fail(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(1)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", required=True)
        args = parser.parse_args()
        run = Path(args.run)
        failures: list[dict] = []

        validation = validate_output_bundle(run_dir=str(run))
        if not validation.ok:
            failures.append({"kind": "validate_output_bundle", "errors": validation.errors})

        v = pl.read_parquet(run / "V_fields.parquet")
        field_ids = set(v["field_id"].to_list())
        required = {
            "sinasc_births_all",
            "sinasc_low_birth_weight_births",
            "sinasc_prematurity_births",
            "sinasc_cesarean_births",
            "sinasc_congenital_anomaly_births",
            "sinasc_low_apgar5_births",
            "sinasc_adolescent_mother_births",
            "sinasc_advanced_maternal_age_births",
            "sinasc_insufficient_prenatal_births",
            "sinasc_low_birth_weight_prevalence",
            "sinasc_prematurity_prevalence",
            "sinasc_cesarean_prevalence",
            "sinasc_congenital_anomaly_prevalence",
            "sinasc_low_apgar5_prevalence",
            "sinasc_adolescent_mother_share",
            "sinasc_advanced_maternal_age_share",
            "sinasc_insufficient_prenatal_share",
        }
        missing = sorted(required - field_ids)
        if missing:
            failures.append({"kind": "missing_fields", "missing": missing})

        failed = pl.read_parquet(run / "FailedBranches.parquet")
        reasons = set(failed["reason"].to_list()) if "reason" in failed.columns else set()
        for reason in [
            "blocked_missing_population_denominator_anchor",
            "blocked_missing_sim_death_numerator_linkage",
            "blocked_missing_sim_neonatal_death_numerator_linkage",
            "blocked_missing_sim_postneonatal_death_numerator_linkage",
        ]:
            if reason not in reasons:
                failures.append({"kind": "missing_failed_branch", "reason": reason})

        warnings = pl.read_parquet(run / "Warnings.parquet")
        messages = "\n".join(str(x) for x in warnings["message"].to_list()) if "message" in warnings.columns else ""
        if "Maternal and newborn administrative race axes remain separate" not in messages:
            failures.append({"kind": "missing_race_axis_warning", "messages": messages})
        if "Crude birth rate requires a legal population denominator" not in messages:
            failures.append({"kind": "missing_birth_rate_denominator_warning", "messages": messages})

        table_path = run / "Tables" / "sinasc_maternal_child_summary.parquet"
        if not table_path.exists():
            failures.append({"kind": "missing_summary_table", "path": str(table_path)})
        else:
            summary = pl.read_parquet(table_path)
            if summary["births_total"].item() <= 0:
                failures.append({"kind": "invalid_birth_count", "births_total": summary["births_total"].item()})
            for col in ["low_birth_weight_prevalence", "prematurity_prevalence", "congenital_anomaly_prevalence"]:
                val = summary[col].item()
                if val is None or val < 0 or val > 1:
                    failures.append({"kind": "invalid_prevalence", "column": col, "value": val})

        if failures:
            fail({"status": "failed", "failures": failures})

        print("AUDIT PASSED: Slice 3A SINASC maternal-child fields validate, preserve race-axis warning, and block mortality/birth-rate denominators until legal linkage exists.")


    if __name__ == "__main__":
        main()
    ''')

    patch_cli()

    print("Applied Slice 3A SINASC maternal-child foundation.")
    print("Run the validation commands supplied by the assistant.")


if __name__ == "__main__":
    main()
