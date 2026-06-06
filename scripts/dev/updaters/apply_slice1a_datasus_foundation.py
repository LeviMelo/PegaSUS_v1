from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/datasus/decoders.py", r'''
    from __future__ import annotations

    import re
    from typing import Any, Literal

    from pydantic import BaseModel, ConfigDict


    class DecodedAge(BaseModel):
        model_config = ConfigDict(extra="forbid")
        age_years: float | None
        age_days: float | None
        age_unit: str | None
        raw_value: str | None
        state: str
        warning: str | None = None


    class DecodedScalar(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: float | None
        unit: str
        raw_value: str | None
        state: str
        warning: str | None = None


    class DecodedCount(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int | None
        raw_value: str | None
        state: str
        warning: str | None = None


    class DecodedBoolean(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int | None
        raw_value: str | None
        state: Literal[
            "ValidFalse",
            "ValidTrue",
            "MissingFlag",
            "InvalidFlagState",
            "UnparseableFlag",
        ]
        warning: str | None = None


    class FilteredCNPJ(BaseModel):
        model_config = ConfigDict(extra="forbid")
        cnpj: str | None
        raw_value: str | None
        state: Literal[
            "ValidCNPJ",
            "NullifiedZeroCNPJ",
            "InvalidCNPJLength",
            "InvalidCNPJDigits",
            "MissingCNPJ",
            "UnparseableCNPJ",
        ]
        warning: str | None = None


    def _none_or_blank(raw: Any) -> bool:
        return raw is None or str(raw).strip() == "" or str(raw).strip().upper() in {"NA", "NAN", "NULL"}


    def decode_sim_idade(raw: str | int | None) -> DecodedAge:
        if _none_or_blank(raw):
            return DecodedAge(
                age_years=None,
                age_days=None,
                age_unit=None,
                raw_value=None if raw is None else str(raw),
                state="UnknownAge",
                warning="missing_age",
            )

        raw_s = str(raw).strip()
        if not raw_s.isdigit():
            return DecodedAge(
                age_years=None,
                age_days=None,
                age_unit=None,
                raw_value=raw_s,
                state="UnknownAge",
                warning="unparseable_age",
            )

        z = raw_s.zfill(3)
        try:
            code = int(z)
        except ValueError:
            return DecodedAge(
                age_years=None,
                age_days=None,
                age_unit=None,
                raw_value=raw_s,
                state="UnknownAge",
                warning="unparseable_age",
            )

        unit = code // 100
        magnitude = code - 100 * unit

        if unit == 1:
            return DecodedAge(
                age_years=magnitude / (24 * 365.25),
                age_days=magnitude / 24,
                age_unit="hours",
                raw_value=raw_s,
                state="valid",
            )
        if unit == 2:
            return DecodedAge(
                age_years=magnitude / 365.25,
                age_days=float(magnitude),
                age_unit="days",
                raw_value=raw_s,
                state="valid",
            )
        if unit == 3:
            return DecodedAge(
                age_years=magnitude / 12,
                age_days=30.4375 * magnitude,
                age_unit="months",
                raw_value=raw_s,
                state="valid",
            )
        if unit == 4:
            return DecodedAge(
                age_years=float(magnitude),
                age_days=365.25 * magnitude,
                age_unit="years",
                raw_value=raw_s,
                state="valid",
            )
        if unit == 5:
            years = 100 + magnitude
            return DecodedAge(
                age_years=float(years),
                age_days=365.25 * years,
                age_unit="years_100_plus",
                raw_value=raw_s,
                state="valid",
            )

        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=raw_s,
            state="UnknownAge",
            warning="unknown_age_unit",
        )


    def decode_sih_age(cod_idade: str | int | None, idade: str | int | None) -> DecodedAge:
        raw_unit = None if cod_idade is None else str(cod_idade).strip()
        raw_age = None if idade is None else str(idade).strip()

        if _none_or_blank(raw_unit):
            return DecodedAge(
                age_years=None,
                age_days=None,
                age_unit=None,
                raw_value=raw_age,
                state="UnknownAgeUnit",
                warning="age_unit_missing",
            )

        if _none_or_blank(raw_age) or not str(raw_age).isdigit():
            return DecodedAge(
                age_years=None,
                age_days=None,
                age_unit=None,
                raw_value=raw_age,
                state="UnknownAge",
                warning="age_missing_or_unparseable",
            )

        m = int(raw_age)

        # microdatasus/SIH codebooks vary by processing layer; keep this registry-replaceable.
        # Conservative defaults: 2=days, 3=months, 4=years, 5=100+ years.
        if raw_unit == "2":
            return DecodedAge(age_years=m / 365.25, age_days=float(m), age_unit="days", raw_value=raw_age, state="valid")
        if raw_unit == "3":
            return DecodedAge(age_years=m / 12, age_days=30.4375 * m, age_unit="months", raw_value=raw_age, state="valid")
        if raw_unit == "4":
            return DecodedAge(age_years=float(m), age_days=365.25 * m, age_unit="years", raw_value=raw_age, state="valid")
        if raw_unit == "5":
            years = 100 + m
            return DecodedAge(age_years=float(years), age_days=365.25 * years, age_unit="years_100_plus", raw_value=raw_age, state="valid")

        return DecodedAge(
            age_years=None,
            age_days=None,
            age_unit=None,
            raw_value=raw_age,
            state="UnknownAgeUnit",
            warning="age_unit_invalid",
        )


    def decode_physical_scalar(
        raw: str | int | float | None,
        *,
        unit: str,
        lower: float,
        upper: float,
        sentinels: set[str],
    ) -> DecodedScalar:
        if _none_or_blank(raw):
            return DecodedScalar(value=None, unit=unit, raw_value=None if raw is None else str(raw), state="MissingScalar", warning="missing_scalar")

        raw_s = str(raw).strip()
        if raw_s in sentinels:
            return DecodedScalar(value=None, unit=unit, raw_value=raw_s, state="InvalidScalar", warning="sentinel_scalar")

        try:
            value = float(raw_s.replace(",", "."))
        except ValueError:
            return DecodedScalar(value=None, unit=unit, raw_value=raw_s, state="UnparseableScalar", warning="unparseable_scalar")

        if value < lower or value > upper:
            return DecodedScalar(value=None, unit=unit, raw_value=raw_s, state="OutOfRangeScalar", warning="out_of_range_scalar")

        return DecodedScalar(value=value, unit=unit, raw_value=raw_s, state="valid")


    def decode_count2(raw: str | int | None, *, sentinels: set[str]) -> DecodedCount:
        if _none_or_blank(raw):
            return DecodedCount(value=None, raw_value=None if raw is None else str(raw), state="UnknownCount", warning="missing_count")

        raw_s = str(raw).strip().zfill(2)
        if raw_s in sentinels:
            return DecodedCount(value=None, raw_value=raw_s, state="UnknownCount", warning="sentinel_count")

        if re.fullmatch(r"\d{1,2}", raw_s):
            return DecodedCount(value=int(raw_s), raw_value=raw_s, state="valid")

        return DecodedCount(value=None, raw_value=raw_s, state="InvalidCount", warning="invalid_count")


    def clamp_bool(raw: object) -> DecodedBoolean:
        if raw is None:
            return DecodedBoolean(value=None, raw_value=None, state="MissingFlag", warning="missing_flag")

        raw_s = str(raw).strip()
        norm = raw_s.casefold()

        if norm in {"", "na", "nan", "null"}:
            return DecodedBoolean(value=None, raw_value=raw_s, state="MissingFlag", warning="missing_flag")

        if norm in {"0", "não", "nao", "no", "false", "f"}:
            return DecodedBoolean(value=0, raw_value=raw_s, state="ValidFalse")

        if norm in {"1", "sim", "yes", "true", "t"}:
            return DecodedBoolean(value=1, raw_value=raw_s, state="ValidTrue")

        try:
            parsed = int(float(raw_s.replace(",", ".")))
        except ValueError:
            return DecodedBoolean(value=None, raw_value=raw_s, state="UnparseableFlag", warning="unparseable_flag")

        if parsed not in {0, 1}:
            return DecodedBoolean(value=None, raw_value=raw_s, state="InvalidFlagState", warning="boolean_flag_outlier")

        return DecodedBoolean(value=parsed, raw_value=raw_s, state="ValidTrue" if parsed == 1 else "ValidFalse")


    def filter_cnpj(raw: object) -> FilteredCNPJ:
        if raw is None:
            return FilteredCNPJ(cnpj=None, raw_value=None, state="MissingCNPJ", warning="missing_cnpj")

        raw_s = str(raw).strip()
        if raw_s == "" or raw_s.upper() in {"NA", "NAN", "NULL"}:
            return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="MissingCNPJ", warning="missing_cnpj")

        digits = re.sub(r"\D", "", raw_s)

        if digits == "":
            return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="UnparseableCNPJ", warning="unparseable_cnpj")

        if set(digits) == {"0"}:
            return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="NullifiedZeroCNPJ", warning="zero_cnpj_nullified")

        if len(digits) != 14:
            return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="InvalidCNPJLength", warning="invalid_cnpj_length")

        if not digits.isdigit():
            return FilteredCNPJ(cnpj=None, raw_value=raw_s, state="InvalidCNPJDigits", warning="invalid_cnpj_digits")

        return FilteredCNPJ(cnpj=digits, raw_value=raw_s, state="ValidCNPJ")
    ''')

    write("src/pegasus/datasus/icd_parser.py", r'''
    from __future__ import annotations

    import re
    from typing import Literal

    from pydantic import BaseModel, ConfigDict


    ICD_RE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]?$")
    ICD_ALLOWED_INITIALS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


    class ICDParseResult(BaseModel):
        model_config = ConfigDict(extra="forbid")

        raw: str | None
        normalized: str | None
        parse_state: Literal["valid", "ill-defined", "blank", "invalid", "unparseable", "missing"]
        topology_role: str
        position: str | None
        source_field: str
        raw_marker: str | None = None
        warnings: list[str]


    def normalize_icd(raw: str | None, *, strip_asterisk: bool = True) -> tuple[str | None, str | None]:
        if raw is None:
            return None, None

        value = str(raw).strip().upper()
        if value == "":
            return "", None

        marker = None
        if strip_asterisk and value.startswith("*"):
            marker = "*"
            value = value[1:].strip()

        value = value.replace(".", "")
        return value, marker


    def parse_icd(
        raw: str | None,
        *,
        topology_role: str,
        source_field: str,
        position: str | None = None,
        strip_asterisk: bool = True,
    ) -> ICDParseResult:
        if raw is None:
            return ICDParseResult(
                raw=None,
                normalized=None,
                parse_state="missing",
                topology_role=topology_role,
                position=position,
                source_field=source_field,
                warnings=["missing_icd"],
            )

        normalized, marker = normalize_icd(raw, strip_asterisk=strip_asterisk)

        if normalized == "":
            return ICDParseResult(
                raw=str(raw),
                normalized=None,
                parse_state="blank",
                topology_role=topology_role,
                position=position,
                source_field=source_field,
                raw_marker=marker,
                warnings=["blank_icd"],
            )

        if normalized is None:
            return ICDParseResult(
                raw=str(raw),
                normalized=None,
                parse_state="missing",
                topology_role=topology_role,
                position=position,
                source_field=source_field,
                raw_marker=marker,
                warnings=["missing_icd"],
            )

        if not ICD_RE.fullmatch(normalized):
            return ICDParseResult(
                raw=str(raw),
                normalized=normalized,
                parse_state="unparseable",
                topology_role=topology_role,
                position=position,
                source_field=source_field,
                raw_marker=marker,
                warnings=["unparseable_icd"],
            )

        if normalized[0] not in ICD_ALLOWED_INITIALS:
            return ICDParseResult(
                raw=str(raw),
                normalized=normalized,
                parse_state="invalid",
                topology_role=topology_role,
                position=position,
                source_field=source_field,
                raw_marker=marker,
                warnings=["invalid_icd_initial"],
            )

        # Conservative quality routing: R-codes are syntactically valid but ill-defined
        # for disease-outcome use unless explicitly allowed by a registry.
        if normalized.startswith("R"):
            return ICDParseResult(
                raw=str(raw),
                normalized=normalized,
                parse_state="ill-defined",
                topology_role=topology_role,
                position=position,
                source_field=source_field,
                raw_marker=marker,
                warnings=["ill_defined_icd"],
            )

        return ICDParseResult(
            raw=str(raw),
            normalized=normalized,
            parse_state="valid",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            raw_marker=marker,
            warnings=[],
        )
    ''')

    write("src/pegasus/datasus/cache.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    from pegasus.core.hashing import content_hash


    class DatasusCache:
        def __init__(self, root: str | Path = "data/cache/datasus/microdatasus") -> None:
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

        def request_hash(self, request: dict[str, Any]) -> str:
            return content_hash(request)

        def request_dir(self, request_hash: str) -> Path:
            path = self.root / request_hash
            path.mkdir(parents=True, exist_ok=True)
            return path

        def write_request(self, request_hash: str, request: dict[str, Any]) -> Path:
            path = self.request_dir(request_hash) / "request.json"
            path.write_text(json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8")
            return path
    ''')

    write("src/pegasus/datasus/subprocess.py", r'''
    from __future__ import annotations

    import json
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    from pegasus.core.hashing import sha256_file
    from pegasus.core.schemas import DATASUSRequestManifest
    from pegasus.datasus.cache import DatasusCache


    class DatasusConfig:
        def __init__(self, rscript_path: str = "Rscript") -> None:
            self.rscript_path = rscript_path


    def _kill_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()


    def _heartbeat_stale(path: Path, timeout_seconds: int) -> bool:
        if not path.exists():
            return False
        return time.time() - path.stat().st_mtime > timeout_seconds


    def fetch_datasus_chunk(
        request: DATASUSRequestManifest,
        *,
        config: DatasusConfig,
        cache: DatasusCache,
        timeout_seconds: int,
        heartbeat_timeout_seconds: int,
    ) -> DATASUSRequestManifest:
        rscript = shutil.which(config.rscript_path) or config.rscript_path
        if shutil.which(config.rscript_path) is None and not Path(config.rscript_path).exists():
            return request.model_copy(update={
                "status": "blocked",
                "exit_code": 41,
                "error_message": f"Rscript not found: {config.rscript_path}",
            })

        script = Path(__file__).parent / "r_scripts" / "fetch_process_microdatasus.R"
        if not script.exists():
            return request.model_copy(update={
                "status": "blocked",
                "exit_code": 11,
                "error_message": f"R bridge script missing: {script}",
            })

        out_dir = Path(request.raw_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        heartbeat_path = Path(request.heartbeat_path)
        stdout_path = Path(request.stdout_path)
        stderr_path = Path(request.stderr_path)

        command = [
            rscript,
            str(script),
            "--system", request.system,
            "--uf", request.uf,
            "--year-start", str(request.year_start),
            "--year-end", str(request.year_end),
            "--out-dir", str(out_dir),
        ]
        if request.month_start is not None:
            command.extend(["--month-start", str(request.month_start)])
        if request.month_end is not None:
            command.extend(["--month-end", str(request.month_end)])

        started = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                elapsed = time.time() - started
                if elapsed > timeout_seconds or _heartbeat_stale(heartbeat_path, heartbeat_timeout_seconds):
                    _kill_process_tree(process)
                    return request.model_copy(update={
                        "status": "timeout",
                        "exit_code": 41,
                        "error_message": "R subprocess timed out or heartbeat became stale.",
                    })
                time.sleep(1.0)

        manifest_path = out_dir / "manifest.json"
        if process.returncode != 0:
            return request.model_copy(update={
                "status": "failed",
                "exit_code": int(process.returncode or 60),
                "error_message": f"R subprocess failed with exit code {process.returncode}.",
            })

        if not manifest_path.exists():
            return request.model_copy(update={
                "status": "failed",
                "exit_code": 50,
                "error_message": "R subprocess returned success but manifest.json is missing.",
            })

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_path = Path(payload.get("raw_path", request.raw_path))
            processed_path = Path(payload.get("processed_path", request.processed_path))
            if not raw_path.exists() or not processed_path.exists():
                return request.model_copy(update={
                    "status": "failed",
                    "exit_code": 50,
                    "error_message": "R manifest exists but raw or processed artifact is missing.",
                })

            return request.model_copy(update={
                "status": "success",
                "exit_code": 0,
                "error_message": None,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "raw_sha256": sha256_file(raw_path),
                "processed_sha256": sha256_file(processed_path),
            })
        except Exception as exc:
            return request.model_copy(update={
                "status": "failed",
                "exit_code": 50,
                "error_message": f"Invalid R manifest: {exc}",
            })
    ''')

    write("src/pegasus/datasus/profile.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    import polars as pl


    def _scan(path: Path) -> pl.LazyFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.scan_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pl.scan_csv(path, infer_schema_length=1000, ignore_errors=True)
        if suffix in {".json", ".ndjson"}:
            return pl.scan_ndjson(path)
        raise ValueError(f"Unsupported profile format: {path}")


    def _top_values(df: pl.DataFrame, column: str, limit: int = 20) -> list[dict[str, Any]]:
        vc = (
            df.select(pl.col(column).cast(pl.Utf8).alias(column))
            .group_by(column)
            .len()
            .sort("len", descending=True)
            .head(limit)
        )
        return vc.to_dicts()


    def profile_table(path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
        path = Path(path)
        lf = _scan(path)
        df = lf.collect()

        profile: dict[str, Any] = {
            "path": str(path),
            "row_count": df.height,
            "column_count": len(df.columns),
            "columns": [],
        }

        for col in df.columns:
            series = df[col]
            non_null = series.drop_nulls()
            col_profile: dict[str, Any] = {
                "column": col,
                "dtype": str(series.dtype),
                "row_count": df.height,
                "missing_count": int(series.null_count()),
                "missing_rate": float(series.null_count() / df.height) if df.height else None,
                "nonblank_count": int(non_null.len()),
                "unique_count": int(series.n_unique()),
                "unique_rate": float(series.n_unique() / df.height) if df.height else None,
                "top_values": _top_values(df, col),
            }

            as_utf8 = series.cast(pl.Utf8, strict=False)
            lengths = as_utf8.str.len_chars()
            col_profile["min_length"] = int(lengths.min()) if lengths.len() and lengths.min() is not None else None
            col_profile["max_length"] = int(lengths.max()) if lengths.len() and lengths.max() is not None else None

            numeric = as_utf8.str.replace(",", ".").cast(pl.Float64, strict=False)
            numeric_valid = numeric.drop_nulls()
            col_profile["numeric_parse_count"] = int(numeric_valid.len())
            col_profile["numeric_parse_rate"] = float(numeric_valid.len() / df.height) if df.height else None
            col_profile["numeric_min"] = float(numeric_valid.min()) if numeric_valid.len() else None
            col_profile["numeric_max"] = float(numeric_valid.max()) if numeric_valid.len() else None

            profile["columns"].append(col_profile)

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

        return profile
    ''')

    write("src/pegasus/datasus/schema_compare.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any


    def compare_profiles(
        raw_profile: dict[str, Any],
        processed_profile: dict[str, Any],
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        raw_cols = {c["column"]: c for c in raw_profile.get("columns", [])}
        proc_cols = {c["column"]: c for c in processed_profile.get("columns", [])}

        raw_names = set(raw_cols)
        proc_names = set(proc_cols)

        common = sorted(raw_names & proc_names)
        changed: list[dict[str, Any]] = []

        for col in common:
            r = raw_cols[col]
            p = proc_cols[col]
            diffs = {}
            for key in ["dtype", "missing_count", "unique_count", "numeric_parse_count"]:
                if r.get(key) != p.get(key):
                    diffs[key] = {"raw": r.get(key), "processed": p.get(key)}
            if diffs:
                changed.append({"column": col, "differences": diffs})

        result = {
            "raw_path": raw_profile.get("path"),
            "processed_path": processed_profile.get("path"),
            "raw_only_columns": sorted(raw_names - proc_names),
            "processed_only_columns": sorted(proc_names - raw_names),
            "common_columns_changed": changed,
        }

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        return result
    ''')

    write("src/pegasus/datasus/r_scripts/fetch_process_microdatasus.R", r'''
    # Slice 1A R boundary skeleton.
    # This script is intentionally conservative: it defines the subprocess contract,
    # heartbeat behavior, and package gate. Full source-specific fetching is implemented
    # after Rscript/microdatasus/read.dbc availability is confirmed.

    args <- commandArgs(trailingOnly = TRUE)

    get_arg <- function(flag, default = NULL) {
      idx <- match(flag, args)
      if (is.na(idx) || idx == length(args)) {
        return(default)
      }
      args[[idx + 1]]
    }

    out_dir <- get_arg("--out-dir", ".")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    heartbeat_path <- file.path(out_dir, "heartbeat.json")
    write_heartbeat <- function(stage, message) {
      payload <- paste0(
        '{"stage":"', stage,
        '","timestamp":"', format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z"),
        '","rows_so_far":null,"message":"', message, '"}'
      )
      writeLines(payload, heartbeat_path, useBytes = TRUE)
    }

    write_heartbeat("fetching", "R boundary started")

    if (!requireNamespace("microdatasus", quietly = TRUE)) {
      write_heartbeat("done", "microdatasus missing")
      quit(status = 30)
    }

    if (!requireNamespace("read.dbc", quietly = TRUE)) {
      write_heartbeat("done", "read.dbc missing")
      quit(status = 31)
    }

    write_heartbeat("done", "Slice 1A boundary reached; full fetch not enabled")
    quit(status = 41)
    ''')

    write("scripts/check_r_microdatasus.R", r'''
    cat("R version:", R.version.string, "\n")
    cat("microdatasus:", requireNamespace("microdatasus", quietly = TRUE), "\n")
    cat("read.dbc:", requireNamespace("read.dbc", quietly = TRUE), "\n")
    ''')

    write("tests/unit/test_datasus_decoders.py", r'''
    from pegasus.datasus.decoders import (
        clamp_bool,
        decode_count2,
        decode_physical_scalar,
        decode_sim_idade,
        filter_cnpj,
    )


    def test_decode_sim_idade_years():
        decoded = decode_sim_idade("474")
        assert decoded.state == "valid"
        assert decoded.age_years == 74
        assert decoded.age_unit == "years"


    def test_clamp_bool_outlier_invalid_not_true():
        decoded = clamp_bool("131")
        assert decoded.value is None
        assert decoded.state == "InvalidFlagState"
        assert decoded.warning == "boolean_flag_outlier"


    def test_filter_cnpj_zero_nullified():
        decoded = filter_cnpj("00000000000000")
        assert decoded.cnpj is None
        assert decoded.state == "NullifiedZeroCNPJ"


    def test_decode_count2_preserves_semantics():
        decoded = decode_count2("03", sentinels={"99"})
        assert decoded.value == 3
        assert decoded.raw_value == "03"


    def test_physical_scalar_sentinel():
        decoded = decode_physical_scalar("9999", unit="grams", lower=300, upper=6500, sentinels={"9999"})
        assert decoded.value is None
        assert decoded.state == "InvalidScalar"
    ''')

    write("tests/unit/test_icd_parser.py", r'''
    from pegasus.datasus.icd_parser import parse_icd


    def test_valid_icd():
        result = parse_icd("A419", topology_role="underlying_cause", source_field="CAUSABAS")
        assert result.parse_state == "valid"
        assert result.normalized == "A419"


    def test_asterisk_preserved_as_marker():
        result = parse_icd("*A419", topology_role="terminal_chain", source_field="LINHAA", position="A")
        assert result.parse_state == "valid"
        assert result.normalized == "A419"
        assert result.raw_marker == "*"


    def test_r_code_is_ill_defined():
        result = parse_icd("R99", topology_role="underlying_cause", source_field="CAUSABAS")
        assert result.parse_state == "ill-defined"


    def test_blank_not_coerced_to_r99():
        result = parse_icd("", topology_role="underlying_cause", source_field="CAUSABAS")
        assert result.parse_state == "blank"
        assert result.normalized is None
    ''')

    print("Applied Slice 1A DATASUS foundation: decoders, ICD parser, profiling, schema comparison, R boundary skeleton, tests.")


if __name__ == "__main__":
    main()