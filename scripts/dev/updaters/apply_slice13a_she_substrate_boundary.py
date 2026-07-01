from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path.cwd()


def write_text(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8", newline="\n")


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def replace_marker_block(text: str, start: str, end: str, block: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(block, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    return text


ZERO_VARIANCE = r'''
"""SHE zero-variance and all-missing exclusion gate.

This module is intentionally source-agnostic. It inspects already-materialized
local source artifacts and classifies columns before they can become substrate
candidates. The gate is conservative: 100% missing columns, constant columns,
identifier/hash/raw payload columns, and structurally unsupported columns are
excluded from analytical substrate candidates and remain auditable only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

NULL_TOKENS: frozenset[str] = frozenset({"", "na", "nan", "null", "none", "missing"})
IDENTIFIER_SUFFIXES: tuple[str, ...] = (
    "_id",
    "_hash",
    "_json",
    "_raw",
    "raw_json",
    "row_hash",
    "raw_record_hash",
    "processed_record_hash",
    "source_manifest_hash",
)
IDENTIFIER_NAMES: frozenset[str] = frozenset({
    "event_id",
    "admission_id",
    "facility_id",
    "raw_json",
    "row_hash",
    "raw_record_hash",
    "processed_record_hash",
    "source_manifest_hash",
})
STATE_MARKERS: tuple[str, ...] = ("_state", "_parse_state", "_states_json", "_state_json")


class ZeroVarianceError(ValueError):
    """Raised when a substrate variance gate cannot inspect an artifact."""


@dataclass(frozen=True)
class ColumnVarianceProfile:
    column: str
    dtype: str
    row_count: int
    non_null_count: int
    missing_count: int
    missing_rate: float | None
    unique_non_null_count: int
    constant_value_repr: str | None
    numeric_parse_count: int
    numeric_min: float | None
    numeric_max: float | None
    structural_role: str
    admissible: bool
    exclusion_reason: str | None
    warnings: tuple[str, ...]

    @property
    def all_missing(self) -> bool:
        return self.row_count > 0 and self.non_null_count == 0

    @property
    def constant(self) -> bool:
        return self.non_null_count > 0 and self.unique_non_null_count <= 1

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        payload["all_missing"] = self.all_missing
        payload["constant"] = self.constant
        return payload


@dataclass(frozen=True)
class TableVarianceProfile:
    path: str
    row_count: int
    column_count: int
    admissible_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    all_missing_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    structural_only_columns: tuple[str, ...]
    profiles: tuple[ColumnVarianceProfile, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "admissible_columns": list(self.admissible_columns),
            "excluded_columns": list(self.excluded_columns),
            "all_missing_columns": list(self.all_missing_columns),
            "constant_columns": list(self.constant_columns),
            "structural_only_columns": list(self.structural_only_columns),
            "profiles": [p.as_manifest() for p in self.profiles],
        }


def _read_table(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if not p.exists():
        raise ZeroVarianceError(f"Source artifact not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(p)
    if suffix in {".csv", ".txt"}:
        return pl.read_csv(p, infer_schema_length=1000, ignore_errors=False)
    if suffix in {".json", ".ndjson"}:
        return pl.read_ndjson(p)
    raise ZeroVarianceError(f"Unsupported substrate artifact format: {p}")


def _blank_normalized_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()


def _non_missing_expr(column: str) -> pl.Expr:
    return pl.col(column).is_not_null() & (~_blank_normalized_expr(column).is_in(list(NULL_TOKENS)))


def classify_structural_role(column: str) -> str:
    name = column.strip()
    lower = name.lower()
    if lower in IDENTIFIER_NAMES or lower.endswith(IDENTIFIER_SUFFIXES):
        return "identifier_or_raw_payload"
    if any(lower.endswith(marker) for marker in STATE_MARKERS):
        return "state_or_quality_marker"
    if lower.startswith("raw_") or lower.endswith("_raw_json"):
        return "raw_payload"
    if lower.endswith("_code") or lower.endswith("_norm") or lower.endswith("_icd_norm"):
        return "categorical_or_code"
    if lower.endswith("_flag") or lower.startswith("is_"):
        return "boolean_measure"
    if any(token in lower for token in ("count", "total", "days", "cost", "weight", "age", "year", "month", "score", "value")):
        return "quantitative_measure"
    return "candidate_measure"


def _constant_repr(values: list[Any]) -> str | None:
    if not values:
        return None
    first = values[0]
    try:
        return json.dumps(first, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(first)


def _numeric_stats(series: pl.Series, non_missing: pl.Series) -> tuple[int, float | None, float | None]:
    if series.len() == 0:
        return 0, None, None
    try:
        numeric = series.cast(pl.Utf8, strict=False).str.replace_all(",", ".").cast(pl.Float64, strict=False)
        numeric = numeric.filter(non_missing)
        numeric_valid = numeric.drop_nulls()
        count = int(numeric_valid.len())
        if count == 0:
            return 0, None, None
        min_v = numeric_valid.min()
        max_v = numeric_valid.max()
        return count, float(min_v) if min_v is not None and math.isfinite(float(min_v)) else None, float(max_v) if max_v is not None and math.isfinite(float(max_v)) else None
    except Exception:
        return 0, None, None


def profile_column(df: pl.DataFrame, column: str) -> ColumnVarianceProfile:
    if column not in df.columns:
        raise ZeroVarianceError(f"Column not found for variance profiling: {column}")
    row_count = int(df.height)
    series = df[column]
    dtype = str(series.dtype)
    if row_count == 0:
        return ColumnVarianceProfile(
            column=column,
            dtype=dtype,
            row_count=0,
            non_null_count=0,
            missing_count=0,
            missing_rate=None,
            unique_non_null_count=0,
            constant_value_repr=None,
            numeric_parse_count=0,
            numeric_min=None,
            numeric_max=None,
            structural_role=classify_structural_role(column),
            admissible=False,
            exclusion_reason="empty_table",
            warnings=("empty_source_artifact",),
        )

    non_missing_mask = df.select(_non_missing_expr(column).alias("non_missing"))["non_missing"]
    non_null_count = int(non_missing_mask.sum())
    missing_count = row_count - non_null_count
    missing_rate = missing_count / float(row_count) if row_count else None
    non_missing_values = series.filter(non_missing_mask).to_list()
    unique_non_null_count = len({json.dumps(v, sort_keys=True, default=str, ensure_ascii=False) for v in non_missing_values})
    structural_role = classify_structural_role(column)
    numeric_parse_count, numeric_min, numeric_max = _numeric_stats(series, non_missing_mask)

    warnings: list[str] = []
    exclusion_reason: str | None = None
    admissible = True

    if non_null_count == 0:
        admissible = False
        exclusion_reason = "all_missing"
        warnings.append("substrate_all_missing_excluded")
    elif unique_non_null_count <= 1:
        admissible = False
        exclusion_reason = "zero_variance_constant"
        warnings.append("substrate_zero_variance_excluded")
    elif structural_role in {"identifier_or_raw_payload", "raw_payload", "state_or_quality_marker"}:
        admissible = False
        exclusion_reason = "structural_or_audit_only"
        warnings.append("substrate_structural_column_excluded")

    return ColumnVarianceProfile(
        column=column,
        dtype=dtype,
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_rate=missing_rate,
        unique_non_null_count=unique_non_null_count,
        constant_value_repr=_constant_repr(non_missing_values) if unique_non_null_count <= 1 else None,
        numeric_parse_count=numeric_parse_count,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
        structural_role=structural_role,
        admissible=admissible,
        exclusion_reason=exclusion_reason,
        warnings=tuple(warnings),
    )


def profile_table_variance(path: str | Path, *, columns: Iterable[str] | None = None) -> TableVarianceProfile:
    p = Path(path)
    df = _read_table(p)
    selected = list(columns) if columns is not None else list(df.columns)
    profiles = tuple(profile_column(df, column) for column in selected if column in df.columns)
    admissible = tuple(p.column for p in profiles if p.admissible)
    excluded = tuple(p.column for p in profiles if not p.admissible)
    all_missing = tuple(p.column for p in profiles if p.exclusion_reason == "all_missing")
    constant = tuple(p.column for p in profiles if p.exclusion_reason == "zero_variance_constant")
    structural = tuple(p.column for p in profiles if p.exclusion_reason == "structural_or_audit_only")
    return TableVarianceProfile(
        path=str(p),
        row_count=int(df.height),
        column_count=len(df.columns),
        admissible_columns=admissible,
        excluded_columns=excluded,
        all_missing_columns=all_missing,
        constant_columns=constant,
        structural_only_columns=structural,
        profiles=profiles,
    )


def assert_no_zero_variance_admissible(profile: TableVarianceProfile) -> None:
    bad = [p.column for p in profile.profiles if p.admissible and (p.all_missing or p.constant)]
    if bad:
        raise AssertionError(f"Zero-variance/all-missing columns were admitted: {bad}")
'''


SOURCE_REGISTRY = r'''
"""Registry-backed source-field semantics for the SHE substrate boundary.

Slice 13A provides a compact built-in registry for already-normalized smoke
artifacts. It is intentionally replaceable by YAML registry loading in the next
registry-realization slice. The registry is used to keep source semantics out of
bundle builders while preserving carrier/unit/aggregation/provenance metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.she.zero_variance import classify_structural_role


class SourceRegistryError(ValueError):
    """Raised when source-field registry resolution fails."""


@dataclass(frozen=True)
class SourceFieldSpec:
    source_system: str
    column: str
    technical_name: str
    carrier: str
    unit: str
    aggregation: str
    role: tuple[str, ...]
    axes: dict[str, Any]
    provenance: tuple[str, ...]
    substrate_kind: str
    admissible_by_registry: bool
    registry_reason: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = list(self.role)
        payload["provenance"] = list(self.provenance)
        return payload


@dataclass(frozen=True)
class SourceRegistryResolution:
    source_system: str
    registry_hash: str
    specs: tuple[SourceFieldSpec, ...]
    unresolved_columns: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "registry_hash": self.registry_hash,
            "specs": [s.as_manifest() for s in self.specs],
            "unresolved_columns": list(self.unresolved_columns),
        }


def _spec(
    source_system: str,
    column: str,
    *,
    carrier: str,
    unit: str,
    aggregation: str,
    role: tuple[str, ...],
    axes: dict[str, Any] | None = None,
    provenance: tuple[str, ...] | None = None,
    substrate_kind: str = "observed_source_field",
    admissible_by_registry: bool = True,
    registry_reason: str | None = None,
) -> SourceFieldSpec:
    return SourceFieldSpec(
        source_system=source_system,
        column=column,
        technical_name=f"{source_system}.{column}",
        carrier=carrier,
        unit=unit,
        aggregation=aggregation,
        role=role,
        axes=axes or {},
        provenance=provenance or ("normalized_source",),
        substrate_kind=substrate_kind,
        admissible_by_registry=admissible_by_registry,
        registry_reason=registry_reason,
    )


BUILTIN_SOURCE_FIELD_REGISTRY: dict[str, dict[str, SourceFieldSpec]] = {
    "SIM-DO": {
        "year": _spec("SIM-DO", "year", carrier="time", unit="year", aggregation="non_aggregable", role=("axis",), axes={"time": "year"}, admissible_by_registry=False, registry_reason="axis_column"),
        "mun_residence_cod6": _spec("SIM-DO", "mun_residence_cod6", carrier="municipality", unit="datasus_cod6", aggregation="non_aggregable", role=("axis",), axes={"geography": "mun_residence_cod6"}, admissible_by_registry=False, registry_reason="axis_column"),
        "age_years": _spec("SIM-DO", "age_years", carrier="deaths", unit="years", aggregation="statistical_functional", role=("covariate", "age"), axes={"age": "continuous_years"}),
        "sex": _spec("SIM-DO", "sex", carrier="deaths", unit="category", aggregation="non_aggregable", role=("axis",), axes={"sex": "administrative"}),
        "race_color_admin": _spec("SIM-DO", "race_color_admin", carrier="deaths", unit="category", aggregation="compositional", role=("axis", "race_admin_raw"), axes={"race_axis_type": "administrative_death_declaration"}),
        "underlying_icd_norm": _spec("SIM-DO", "underlying_icd_norm", carrier="deaths", unit="ICD10", aggregation="additive", role=("health_seed", "diagnostic_topology"), axes={"diagnostic_role": "underlying_cause"}),
        "associated_conditions_norm": _spec("SIM-DO", "associated_conditions_norm", carrier="deaths", unit="ICD10", aggregation="additive", role=("observer", "diagnostic_topology"), axes={"diagnostic_role": "associated_condition"}),
        "reporting_delay": _spec("SIM-DO", "reporting_delay", carrier="deaths", unit="days", aggregation="statistical_functional", role=("institutional_quality",)),
    },
    "SINASC": {
        "birth_year": _spec("SINASC", "birth_year", carrier="time", unit="year", aggregation="non_aggregable", role=("axis",), axes={"time": "year"}, admissible_by_registry=False, registry_reason="axis_column"),
        "mun_residence_cod6": _spec("SINASC", "mun_residence_cod6", carrier="live_births", unit="datasus_cod6", aggregation="non_aggregable", role=("axis",), axes={"geography": "mun_residence_cod6"}, admissible_by_registry=False, registry_reason="axis_column"),
        "birth_weight_g": _spec("SINASC", "birth_weight_g", carrier="live_births", unit="grams", aggregation="statistical_functional", role=("child_outcome",)),
        "low_birth_weight_flag": _spec("SINASC", "low_birth_weight_flag", carrier="live_births", unit="count", aggregation="additive", role=("child_outcome", "binary_event")),
        "prematurity_flag": _spec("SINASC", "prematurity_flag", carrier="live_births", unit="count", aggregation="additive", role=("child_outcome", "binary_event")),
        "cesarean_flag": _spec("SINASC", "cesarean_flag", carrier="live_births", unit="count", aggregation="additive", role=("delivery", "binary_event")),
        "congenital_anomaly_flag": _spec("SINASC", "congenital_anomaly_flag", carrier="live_births", unit="count", aggregation="additive", role=("congenital_anomaly", "binary_event")),
        "anomaly_icd_code": _spec("SINASC", "anomaly_icd_code", carrier="live_births", unit="ICD10", aggregation="additive", role=("congenital_anomaly", "diagnostic_topology"), axes={"diagnostic_role": "birth_anomaly"}),
    },
    "SIH-RD": {
        "admission_year": _spec("SIH-RD", "admission_year", carrier="time", unit="year", aggregation="non_aggregable", role=("axis",), axes={"time": "year"}, admissible_by_registry=False, registry_reason="axis_column"),
        "mun_residence_cod6": _spec("SIH-RD", "mun_residence_cod6", carrier="municipality", unit="datasus_cod6", aggregation="non_aggregable", role=("axis",), axes={"geography": "mun_residence_cod6"}, admissible_by_registry=False, registry_reason="axis_column"),
        "principal_icd_norm": _spec("SIH-RD", "principal_icd_norm", carrier="hospitalizations", unit="ICD10", aggregation="additive", role=("hospital_principal_diagnosis", "diagnostic_topology"), axes={"diagnostic_role": "sih_principal_diagnosis"}),
        "stay_length_days": _spec("SIH-RD", "stay_length_days", carrier="hospitalizations", unit="days", aggregation="statistical_functional", role=("length_of_stay",)),
        "death_flag": _spec("SIH-RD", "death_flag", carrier="hospitalizations", unit="count", aggregation="additive", role=("inpatient_death", "binary_event")),
        "hospital_service_cost_real": _spec("SIH-RD", "hospital_service_cost_real", carrier="hospitalizations", unit="BRL", aggregation="additive", role=("sih_cost_component", "VAL_SH"), axes={"sih_cost_component": "VAL_SH"}),
        "professional_service_cost_real": _spec("SIH-RD", "professional_service_cost_real", carrier="hospitalizations", unit="BRL", aggregation="additive", role=("sih_cost_component", "VAL_SP"), axes={"sih_cost_component": "VAL_SP"}),
        "icu_cost_real": _spec("SIH-RD", "icu_cost_real", carrier="hospitalizations", unit="BRL", aggregation="additive", role=("sih_cost_component", "VAL_UTI"), axes={"sih_cost_component": "VAL_UTI"}),
        "total_admission_cost_real": _spec("SIH-RD", "total_admission_cost_real", carrier="hospitalizations", unit="BRL", aggregation="additive", role=("sih_cost_component", "VAL_TOT"), axes={"sih_cost_component": "VAL_TOT"}),
    },
    "CNES-ST": {
        "year": _spec("CNES-ST", "year", carrier="time", unit="year", aggregation="non_aggregable", role=("axis",), axes={"time": "year"}, admissible_by_registry=False, registry_reason="axis_column"),
        "mun_facility_cod6": _spec("CNES-ST", "mun_facility_cod6", carrier="municipality", unit="datasus_cod6", aggregation="non_aggregable", role=("axis",), axes={"geography": "mun_facility_cod6"}, admissible_by_registry=False, registry_reason="axis_column"),
        "capacity_total_observed": _spec("CNES-ST", "capacity_total_observed", carrier="facilities", unit="capacity_count", aggregation="additive", role=("cnes_capacity_vector_observer",)),
        "invalid_flag_count": _spec("CNES-ST", "invalid_flag_count", carrier="facilities", unit="count", aggregation="additive", role=("cnes_boolean_outlier_observer",)),
    },
    "SIDRA": {
        "value_numeric": _spec("SIDRA", "value_numeric", carrier="context_cube", unit="table_defined", aggregation="weighted_mean", role=("context", "sidra_long_fact"), provenance=("sidra_official",)),
        "period": _spec("SIDRA", "period", carrier="time", unit="period", aggregation="non_aggregable", role=("axis",), axes={"time": "period"}, admissible_by_registry=False, registry_reason="axis_column"),
        "locality_id": _spec("SIDRA", "locality_id", carrier="municipality", unit="IBGE", aggregation="non_aggregable", role=("axis",), axes={"geography": "municipality_ibge_cod7"}, admissible_by_registry=False, registry_reason="axis_column"),
    },
}


def source_registry_hash(source_system: str | None = None) -> str:
    if source_system is None:
        payload = {system: {k: v.as_manifest() for k, v in specs.items()} for system, specs in BUILTIN_SOURCE_FIELD_REGISTRY.items()}
    else:
        payload = {source_system: {k: v.as_manifest() for k, v in BUILTIN_SOURCE_FIELD_REGISTRY.get(source_system, {}).items()}}
    return content_hash(payload)


def _heuristic_spec(source_system: str, column: str) -> SourceFieldSpec:
    role = classify_structural_role(column)
    admissible = role not in {"identifier_or_raw_payload", "raw_payload", "state_or_quality_marker"}
    if role == "quantitative_measure":
        carrier = "source_records"
        unit = "numeric"
        aggregation = "statistical_functional"
    elif role == "boolean_measure":
        carrier = "source_records"
        unit = "count"
        aggregation = "additive"
    elif role == "categorical_or_code":
        carrier = "source_records"
        unit = "category"
        aggregation = "compositional"
    else:
        carrier = "source_records"
        unit = "unknown"
        aggregation = "non_aggregable"
    return _spec(
        source_system,
        column,
        carrier=carrier,
        unit=unit,
        aggregation=aggregation,
        role=("registry_inferred", role),
        axes={},
        provenance=("normalized_source", "heuristic_registry"),
        substrate_kind="observed_source_field",
        admissible_by_registry=admissible,
        registry_reason=None if admissible else "structural_or_audit_only",
    )


def resolve_source_fields(*, source_system: str, columns: list[str], allow_heuristic: bool = True) -> SourceRegistryResolution:
    registry = BUILTIN_SOURCE_FIELD_REGISTRY.get(source_system, {})
    specs: list[SourceFieldSpec] = []
    unresolved: list[str] = []
    for column in columns:
        if column in registry:
            specs.append(registry[column])
        elif allow_heuristic:
            specs.append(_heuristic_spec(source_system, column))
        else:
            unresolved.append(column)
    return SourceRegistryResolution(
        source_system=source_system,
        registry_hash=source_registry_hash(source_system),
        specs=tuple(specs),
        unresolved_columns=tuple(unresolved),
    )


def source_system_from_path(path: str | Path) -> str | None:
    text = str(path).replace("\\", "/").upper()
    for system in BUILTIN_SOURCE_FIELD_REGISTRY:
        if system.upper() in text:
            return system
    if "SIM" in text:
        return "SIM-DO"
    if "SINASC" in text:
        return "SINASC"
    if "SIH" in text:
        return "SIH-RD"
    if "CNES" in text:
        return "CNES-ST"
    if "SIDRA" in text:
        return "SIDRA"
    return None


def write_registry_manifest(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "registry_kind": "builtin_source_field_registry_slice13a",
        "registry_hash": source_registry_hash(),
        "systems": {
            system: {column: spec.as_manifest() for column, spec in specs.items()}
            for system, specs in BUILTIN_SOURCE_FIELD_REGISTRY.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
'''


SUBSTRATE = r'''
"""Substrate Harmonization Engine boundary for source artifacts.

Slice 13A introduces the first real SHE boundary. It consumes local source
artifacts, applies source-field registry semantics and zero-variance/all-missing
exclusion, and emits a typed SubstrateBundle. It does not build the EFG, compute
rates, fetch data, run PIRS/HSIC, or materialize first-class output keys.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from pegasus.core.hashing import content_hash, sha256_file
from pegasus.she.source_registry import SourceFieldSpec, resolve_source_fields, source_system_from_path
from pegasus.she.zero_variance import ColumnVarianceProfile, TableVarianceProfile, profile_table_variance


class SubstrateError(ValueError):
    """Raised when the SHE substrate boundary cannot evaluate an artifact."""


@dataclass(frozen=True)
class SourceArtifactRef:
    path: str
    source_system: str
    artifact_role: str
    provenance_mode: str
    source_manifest_hash: str | None = None
    artifact_hash: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubstrateFieldCandidate:
    candidate_id: str
    source_system: str
    artifact_path: str
    column: str
    technical_name: str
    carrier: str
    unit: str
    aggregation: str
    role: tuple[str, ...]
    axes: dict[str, Any]
    provenance: tuple[str, ...]
    substrate_kind: str
    registry_hash: str
    row_count: int
    non_null_count: int
    unique_non_null_count: int
    missing_rate: float | None
    numeric_min: float | None
    numeric_max: float | None
    source_manifest_hash: str | None
    artifact_hash: str | None
    warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["role"] = list(self.role)
        payload["provenance"] = list(self.provenance)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class SubstrateFieldExclusion:
    exclusion_id: str
    source_system: str
    artifact_path: str
    column: str
    reason: str
    registry_reason: str | None
    row_count: int
    non_null_count: int
    unique_non_null_count: int
    missing_rate: float | None
    structural_role: str
    warnings: tuple[str, ...]

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True)
class SubstrateBundle:
    schema_version: str
    substrate_id: str
    source_reality_mode: str
    source_artifacts: tuple[SourceArtifactRef, ...]
    candidates: tuple[SubstrateFieldCandidate, ...]
    exclusions: tuple[SubstrateFieldExclusion, ...]
    table_profiles: tuple[TableVarianceProfile, ...]
    registry_hashes: dict[str, str]
    warnings: tuple[str, ...]

    @property
    def admissible_candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def excluded_field_count(self) -> int:
        return len(self.exclusions)

    @property
    def zero_variance_exclusion_count(self) -> int:
        return sum(1 for e in self.exclusions if e.reason == "zero_variance_constant")

    @property
    def all_missing_exclusion_count(self) -> int:
        return sum(1 for e in self.exclusions if e.reason == "all_missing")

    @property
    def structural_exclusion_count(self) -> int:
        return sum(1 for e in self.exclusions if e.reason == "structural_or_audit_only")

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "substrate_id": self.substrate_id,
            "source_reality_mode": self.source_reality_mode,
            "source_artifact_count": len(self.source_artifacts),
            "admissible_candidate_count": self.admissible_candidate_count,
            "excluded_field_count": self.excluded_field_count,
            "zero_variance_exclusion_count": self.zero_variance_exclusion_count,
            "all_missing_exclusion_count": self.all_missing_exclusion_count,
            "structural_exclusion_count": self.structural_exclusion_count,
            "registry_hashes": dict(self.registry_hashes),
            "warnings": list(self.warnings),
        }

    def as_manifest(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "source_artifacts": [a.as_manifest() for a in self.source_artifacts],
            "candidates": [c.as_manifest() for c in self.candidates],
            "exclusions": [e.as_manifest() for e in self.exclusions],
            "table_profiles": [p.as_manifest() for p in self.table_profiles],
        }


def _artifact_hash(path: str | Path) -> str | None:
    p = Path(path)
    return sha256_file(p) if p.exists() and p.is_file() else None


def _stable_id(payload: dict[str, Any], prefix: str) -> str:
    return f"{prefix}_{content_hash(payload)[:20]}"


def normalize_source_artifact_ref(payload: SourceArtifactRef | dict[str, Any] | str | Path) -> SourceArtifactRef:
    if isinstance(payload, SourceArtifactRef):
        return payload
    if isinstance(payload, (str, Path)):
        path = str(payload)
        source_system = source_system_from_path(path) or "UNKNOWN"
        return SourceArtifactRef(
            path=path,
            source_system=source_system,
            artifact_role="processed_events",
            provenance_mode="fixture" if "fixture" in path.lower() else "cached_external",
            artifact_hash=_artifact_hash(path),
        )
    if not isinstance(payload, dict):
        raise SubstrateError(f"Invalid source artifact reference: {payload!r}")
    path = payload.get("path") or payload.get("artifact_path") or payload.get("processed_path")
    if not path:
        raise SubstrateError(f"Source artifact reference has no path: {payload}")
    source_system = payload.get("source_system") or payload.get("system") or source_system_from_path(path) or "UNKNOWN"
    role = payload.get("artifact_role") or payload.get("role") or payload.get("source_role") or "processed_events"
    mode = payload.get("provenance_mode") or payload.get("mode") or payload.get("source_mode") or ("fixture" if "fixture" in str(path).lower() else "cached_external")
    source_manifest_hash = payload.get("source_manifest_hash") or payload.get("manifest_hash")
    artifact_hash = payload.get("artifact_hash") or payload.get("sha256") or _artifact_hash(path)
    return SourceArtifactRef(
        path=str(path),
        source_system=str(source_system),
        artifact_role=str(role),
        provenance_mode=str(mode),
        source_manifest_hash=None if source_manifest_hash is None else str(source_manifest_hash),
        artifact_hash=None if artifact_hash is None else str(artifact_hash),
    )


def load_source_artifacts_from_manifest(path: str | Path) -> tuple[SourceArtifactRef, ...]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw_artifacts = payload.get("artifacts") or payload.get("source_artifacts") or []
    if isinstance(raw_artifacts, dict):
        raw_artifacts = list(raw_artifacts.values())
    if not isinstance(raw_artifacts, list):
        raise SubstrateError(f"Source artifact manifest does not contain a list of artifacts: {p}")
    return tuple(normalize_source_artifact_ref(x) for x in raw_artifacts)


def source_reality_mode(artifacts: Iterable[SourceArtifactRef]) -> str:
    modes = {a.provenance_mode for a in artifacts}
    if not modes:
        return "fixture_only"
    if modes == {"materialized_external"}:
        return "materialized_external"
    if modes <= {"cached_external", "materialized_external"}:
        return "cached_external"
    if modes == {"fixture"}:
        return "fixture_only"
    return "mixed_fixture_external"


def _candidate_from_profile(
    *,
    artifact: SourceArtifactRef,
    spec: SourceFieldSpec,
    registry_hash: str,
    profile: ColumnVarianceProfile,
) -> SubstrateFieldCandidate:
    payload = {
        "source_system": artifact.source_system,
        "path": artifact.path,
        "column": profile.column,
        "registry_hash": registry_hash,
        "artifact_hash": artifact.artifact_hash,
    }
    warnings = tuple(sorted(set(profile.warnings)))
    return SubstrateFieldCandidate(
        candidate_id=_stable_id(payload, "substrate_candidate"),
        source_system=artifact.source_system,
        artifact_path=artifact.path,
        column=profile.column,
        technical_name=spec.technical_name,
        carrier=spec.carrier,
        unit=spec.unit,
        aggregation=spec.aggregation,
        role=spec.role,
        axes=spec.axes,
        provenance=tuple(dict.fromkeys((*spec.provenance, artifact.provenance_mode))),
        substrate_kind=spec.substrate_kind,
        registry_hash=registry_hash,
        row_count=profile.row_count,
        non_null_count=profile.non_null_count,
        unique_non_null_count=profile.unique_non_null_count,
        missing_rate=profile.missing_rate,
        numeric_min=profile.numeric_min,
        numeric_max=profile.numeric_max,
        source_manifest_hash=artifact.source_manifest_hash,
        artifact_hash=artifact.artifact_hash,
        warnings=warnings,
    )


def _exclusion_from_profile(
    *,
    artifact: SourceArtifactRef,
    spec: SourceFieldSpec,
    profile: ColumnVarianceProfile,
    reason: str,
) -> SubstrateFieldExclusion:
    payload = {
        "source_system": artifact.source_system,
        "path": artifact.path,
        "column": profile.column,
        "reason": reason,
        "artifact_hash": artifact.artifact_hash,
    }
    warnings = list(profile.warnings)
    if spec.registry_reason:
        warnings.append(f"registry_{spec.registry_reason}")
    return SubstrateFieldExclusion(
        exclusion_id=_stable_id(payload, "substrate_exclusion"),
        source_system=artifact.source_system,
        artifact_path=artifact.path,
        column=profile.column,
        reason=reason,
        registry_reason=spec.registry_reason,
        row_count=profile.row_count,
        non_null_count=profile.non_null_count,
        unique_non_null_count=profile.unique_non_null_count,
        missing_rate=profile.missing_rate,
        structural_role=profile.structural_role,
        warnings=tuple(sorted(set(warnings))),
    )


def build_substrate_bundle(
    *,
    artifacts: Iterable[SourceArtifactRef | dict[str, Any] | str | Path],
    allow_heuristic_registry: bool = True,
) -> SubstrateBundle:
    refs = tuple(normalize_source_artifact_ref(a) for a in artifacts)
    candidates: list[SubstrateFieldCandidate] = []
    exclusions: list[SubstrateFieldExclusion] = []
    profiles: list[TableVarianceProfile] = []
    registry_hashes: dict[str, str] = {}
    warnings: list[str] = []

    if not refs:
        warnings.append("substrate_no_source_artifacts")

    for artifact in refs:
        path = Path(artifact.path)
        if not path.exists():
            exclusions.append(SubstrateFieldExclusion(
                exclusion_id=_stable_id({"path": artifact.path, "reason": "missing_artifact"}, "substrate_exclusion"),
                source_system=artifact.source_system,
                artifact_path=artifact.path,
                column="*",
                reason="missing_artifact",
                registry_reason=None,
                row_count=0,
                non_null_count=0,
                unique_non_null_count=0,
                missing_rate=None,
                structural_role="missing_artifact",
                warnings=("substrate_artifact_missing",),
            ))
            warnings.append(f"missing_artifact:{artifact.path}")
            continue
        table_profile = profile_table_variance(path)
        profiles.append(table_profile)
        resolution = resolve_source_fields(
            source_system=artifact.source_system,
            columns=[p.column for p in table_profile.profiles],
            allow_heuristic=allow_heuristic_registry,
        )
        registry_hashes[artifact.source_system] = resolution.registry_hash
        spec_by_column = {s.column: s for s in resolution.specs}
        for profile in table_profile.profiles:
            spec = spec_by_column.get(profile.column)
            if spec is None:
                exclusions.append(_exclusion_from_profile(
                    artifact=artifact,
                    spec=SourceFieldSpec(
                        source_system=artifact.source_system,
                        column=profile.column,
                        technical_name=f"{artifact.source_system}.{profile.column}",
                        carrier="unknown",
                        unit="unknown",
                        aggregation="non_aggregable",
                        role=("unresolved",),
                        axes={},
                        provenance=("unresolved_registry",),
                        substrate_kind="unresolved",
                        admissible_by_registry=False,
                        registry_reason="unresolved_source_field",
                    ),
                    profile=profile,
                    reason="unresolved_source_field",
                ))
                continue
            reason = profile.exclusion_reason or spec.registry_reason
            if profile.admissible and spec.admissible_by_registry:
                candidates.append(_candidate_from_profile(
                    artifact=artifact,
                    spec=spec,
                    registry_hash=resolution.registry_hash,
                    profile=profile,
                ))
            else:
                exclusions.append(_exclusion_from_profile(
                    artifact=artifact,
                    spec=spec,
                    profile=profile,
                    reason=reason or "registry_excluded",
                ))

    substrate_payload = {
        "source_reality_mode": source_reality_mode(refs),
        "artifacts": [a.as_manifest() for a in refs],
        "candidate_ids": [c.candidate_id for c in candidates],
        "exclusion_ids": [e.exclusion_id for e in exclusions],
        "registry_hashes": registry_hashes,
    }
    return SubstrateBundle(
        schema_version="1.0",
        substrate_id=_stable_id(substrate_payload, "substrate_bundle"),
        source_reality_mode=source_reality_mode(refs),
        source_artifacts=refs,
        candidates=tuple(candidates),
        exclusions=tuple(exclusions),
        table_profiles=tuple(profiles),
        registry_hashes=registry_hashes,
        warnings=tuple(sorted(set(warnings))),
    )


def write_substrate_bundle_manifest(bundle: SubstrateBundle, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle.as_manifest(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_substrate_bundle_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SubstrateError(f"Substrate manifest is not a JSON object: {path}")
    return payload


def attach_substrate_summary_to_run(*, run_dir: str | Path, bundle: SubstrateBundle) -> dict[str, Any]:
    root = Path(run_dir)
    tables_dir = root / "Tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_substrate_bundle_manifest(bundle, tables_dir / "substrate_manifest.json")
    summary = {
        **bundle.summary(),
        "manifest_path": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "status": "evaluated",
    }
    for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
        except Exception:
            continue
        payload["substrate_gate"] = summary
        if name == "ReproducibilityManifest.json":
            payload.setdefault("registry_hashes", {}).update(bundle.registry_hashes)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
'''


BUILD_SUBSTRATE = r'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.she.substrate import (
    SourceArtifactRef,
    attach_substrate_summary_to_run,
    build_substrate_bundle,
    load_source_artifacts_from_manifest,
    write_substrate_bundle_manifest,
)


def run_build_substrate_from_artifacts(
    *,
    artifacts: list[dict[str, Any]] | list[str],
    output: str | Path | None = None,
) -> dict[str, Any]:
    bundle = build_substrate_bundle(artifacts=artifacts)
    if output is not None:
        write_substrate_bundle_manifest(bundle, output)
    return bundle.as_manifest()


def run_build_substrate_from_source_manifest(
    *,
    source_manifest: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    artifacts = load_source_artifacts_from_manifest(source_manifest)
    bundle = build_substrate_bundle(artifacts=artifacts)
    if output is not None:
        write_substrate_bundle_manifest(bundle, output)
    return bundle.as_manifest()


def run_attach_substrate_to_run(
    *,
    run_dir: str | Path,
    source_manifest: str | Path | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if source_manifest is not None:
        refs = load_source_artifacts_from_manifest(source_manifest)
    else:
        refs = [SourceArtifactRef(**artifact) for artifact in (artifacts or [])]
    bundle = build_substrate_bundle(artifacts=refs)
    return attach_substrate_summary_to_run(run_dir=run_dir, bundle=bundle)


def run_substrate_summary(*, manifest: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return {
        "schema_version": payload.get("schema_version"),
        "substrate_id": payload.get("substrate_id"),
        "source_reality_mode": payload.get("source_reality_mode"),
        "source_artifact_count": payload.get("source_artifact_count"),
        "admissible_candidate_count": payload.get("admissible_candidate_count"),
        "excluded_field_count": payload.get("excluded_field_count"),
        "zero_variance_exclusion_count": payload.get("zero_variance_exclusion_count"),
        "all_missing_exclusion_count": payload.get("all_missing_exclusion_count"),
        "structural_exclusion_count": payload.get("structural_exclusion_count"),
        "warnings": payload.get("warnings", []),
    }
'''


AUDIT = r'''
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Slice 13A SHE substrate boundary artifacts.")
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--run", type=Path, required=False)
    parser.add_argument("--require-exclusions", action="store_true")
    args = parser.parse_args()

    if args.manifest is None and args.run is None:
        raise SystemExit("Provide --manifest or --run.")

    if args.manifest is not None:
        payload = load(args.manifest)
    else:
        path = args.run / "Tables" / "substrate_manifest.json"
        if not path.exists():
            raise SystemExit(f"substrate manifest missing from run: {path}")
        payload = load(path)

    errors: list[str] = []
    if payload.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if not payload.get("substrate_id"):
        errors.append("substrate_id missing")
    if payload.get("admissible_candidate_count", 0) < 0:
        errors.append("admissible_candidate_count invalid")
    if payload.get("excluded_field_count", 0) < 0:
        errors.append("excluded_field_count invalid")
    if args.require_exclusions and payload.get("excluded_field_count", 0) <= 0:
        errors.append("expected at least one substrate exclusion")

    for candidate in payload.get("candidates", []):
        if candidate.get("column") in payload.get("all_missing_columns", []):
            errors.append(f"all-missing column admitted: {candidate.get('column')}")
        if not candidate.get("registry_hash"):
            errors.append(f"candidate missing registry_hash: {candidate.get('column')}")

    for exclusion in payload.get("exclusions", []):
        reason = exclusion.get("reason")
        if reason in {"all_missing", "zero_variance_constant"} and exclusion.get("column") == "*":
            errors.append("wildcard zero-variance/all-missing exclusion is invalid")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, sort_keys=True))
        return 1

    print(
        "AUDIT PASSED: Slice 13A SHE substrate boundary validated "
        f"candidates={payload.get('admissible_candidate_count')} "
        f"exclusions={payload.get('excluded_field_count')} "
        f"zero_variance={payload.get('zero_variance_exclusion_count')} "
        f"all_missing={payload.get('all_missing_exclusion_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


UNIT_TEST_ZERO = r'''
from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.she.zero_variance import profile_table_variance


def test_slice13a_zero_variance_gate_excludes_all_missing_and_constant(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    pl.DataFrame({
        "event_id": ["a", "b", "c"],
        "all_missing": [None, None, None],
        "constant_cost": [0, 0, 0],
        "variable_cost": [1, 2, 3],
        "status_state": ["valid", "valid", "invalid"],
    }).write_parquet(path)

    profile = profile_table_variance(path)
    reasons = {p.column: p.exclusion_reason for p in profile.profiles}

    assert reasons["all_missing"] == "all_missing"
    assert reasons["constant_cost"] == "zero_variance_constant"
    assert reasons["event_id"] == "structural_or_audit_only"
    assert reasons["status_state"] == "structural_or_audit_only"
    assert "variable_cost" in profile.admissible_columns
    assert "constant_cost" not in profile.admissible_columns
'''


UNIT_TEST_SUBSTRATE = r'''
from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.she.substrate import build_substrate_bundle
from pegasus.she.source_registry import resolve_source_fields


def test_slice13a_source_registry_resolves_sim_semantics() -> None:
    resolution = resolve_source_fields(
        source_system="SIM-DO",
        columns=["underlying_icd_norm", "age_years", "raw_json"],
    )
    specs = {spec.column: spec for spec in resolution.specs}
    assert specs["underlying_icd_norm"].unit == "ICD10"
    assert "diagnostic_topology" in specs["underlying_icd_norm"].role
    assert specs["age_years"].carrier == "deaths"
    assert specs["raw_json"].admissible_by_registry is False


def test_slice13a_substrate_bundle_admits_only_registry_and_variance_valid_fields(tmp_path: Path) -> None:
    path = tmp_path / "sim_events.parquet"
    pl.DataFrame({
        "event_id": ["a", "b", "c"],
        "year": [2020, 2020, 2020],
        "age_years": [10.0, 30.0, 50.0],
        "underlying_icd_norm": ["I10", "J18", "I10"],
        "constant_marker": [1, 1, 1],
        "all_missing_marker": [None, None, None],
    }).write_parquet(path)

    bundle = build_substrate_bundle(artifacts=[{
        "path": str(path),
        "source_system": "SIM-DO",
        "artifact_role": "processed_events",
        "provenance_mode": "fixture",
        "source_manifest_hash": "fixture_manifest_hash",
    }])

    candidate_columns = {c.column for c in bundle.candidates}
    exclusion_reasons = {e.column: e.reason for e in bundle.exclusions}

    assert "age_years" in candidate_columns
    assert "underlying_icd_norm" in candidate_columns
    assert exclusion_reasons["event_id"] == "structural_or_audit_only"
    assert exclusion_reasons["year"] == "zero_variance_constant"
    assert exclusion_reasons["constant_marker"] == "zero_variance_constant"
    assert exclusion_reasons["all_missing_marker"] == "all_missing"
    assert bundle.zero_variance_exclusion_count >= 2
    assert bundle.all_missing_exclusion_count == 1
'''


INTEGRATION_TEST = r'''
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.workflows.construct.build_substrate import (
    run_attach_substrate_to_run,
    run_build_substrate_from_artifacts,
    run_substrate_summary,
)


def test_slice13a_substrate_workflow_writes_manifest_and_summary(tmp_path: Path) -> None:
    events = tmp_path / "sim_events.parquet"
    out = tmp_path / "substrate.json"
    pl.DataFrame({
        "event_id": ["a", "b", "c", "d"],
        "age_years": [20.0, 21.0, 22.0, 23.0],
        "constant_cost": [0, 0, 0, 0],
        "all_missing_cost": [None, None, None, None],
        "underlying_icd_norm": ["I10", "I10", "J18", "J18"],
    }).write_parquet(events)

    payload = run_build_substrate_from_artifacts(
        artifacts=[{
            "path": str(events),
            "source_system": "SIM-DO",
            "artifact_role": "processed_events",
            "provenance_mode": "fixture",
            "source_manifest_hash": "fixture_manifest_hash",
        }],
        output=out,
    )

    assert out.exists()
    assert payload["source_reality_mode"] == "fixture_only"
    assert payload["admissible_candidate_count"] >= 2
    assert payload["zero_variance_exclusion_count"] >= 1
    assert payload["all_missing_exclusion_count"] == 1
    summary = run_substrate_summary(manifest=out)
    assert summary["substrate_id"] == payload["substrate_id"]


def test_slice13a_attach_substrate_metadata_to_valid_run_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    events = tmp_path / "sih_events.parquet"
    create_empty_output_bundle(run_dir)
    pl.DataFrame({
        "admission_id": ["x", "y", "z"],
        "principal_icd_norm": ["G43", "G43", "I10"],
        "stay_length_days": [1, 2, 3],
        "total_admission_cost_real": [100.0, 120.0, 150.0],
        "zero_val_sadt": [0, 0, 0],
    }).write_parquet(events)

    summary = run_attach_substrate_to_run(
        run_dir=run_dir,
        artifacts=[{
            "path": str(events),
            "source_system": "SIH-RD",
            "artifact_role": "processed_events",
            "provenance_mode": "fixture",
            "source_manifest_hash": "fixture_manifest_hash",
        }],
    )

    assert summary["status"] == "evaluated"
    assert summary["admissible_candidate_count"] >= 3
    assert summary["zero_variance_exclusion_count"] >= 1
    assert (run_dir / "Tables" / "substrate_manifest.json").exists()
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["substrate_gate"]["substrate_id"] == summary["substrate_id"]
'''


# Write modules.
write_text("src/pegasus/she/zero_variance.py", ZERO_VARIANCE)
write_text("src/pegasus/she/source_registry.py", SOURCE_REGISTRY)
write_text("src/pegasus/she/substrate.py", SUBSTRATE)
write_text("src/pegasus/workflows/build_substrate.py", BUILD_SUBSTRATE)
write_text("tests/unit/test_she_zero_variance_substrate.py", UNIT_TEST_ZERO)
write_text("tests/unit/test_she_substrate_registry.py", UNIT_TEST_SUBSTRATE)
write_text("tests/integration/test_slice13a_she_substrate_boundary.py", INTEGRATION_TEST)
write_text("scripts/dev/audits/audit_slice13a_she_substrate_boundary.py", AUDIT)

# Patch she/__init__.py exports.
write_text("src/pegasus/she/__init__.py", r'''
"""PegaSUS Substrate Harmonization Engine package."""

from pegasus.she.substrate import (
    SourceArtifactRef,
    SubstrateBundle,
    SubstrateFieldCandidate,
    SubstrateFieldExclusion,
    build_substrate_bundle,
)
from pegasus.she.zero_variance import ColumnVarianceProfile, TableVarianceProfile, profile_table_variance

__all__ = [
    "SourceArtifactRef",
    "SubstrateBundle",
    "SubstrateFieldCandidate",
    "SubstrateFieldExclusion",
    "build_substrate_bundle",
    "ColumnVarianceProfile",
    "TableVarianceProfile",
    "profile_table_variance",
]
''')

# Patch compile.py with a safe wrapper around the existing run_compile.
compile_path = ROOT / "src/pegasus/workflows/compile.py"
compile_text = compile_path.read_text(encoding="utf-8")
start = "# ---- Slice 13A SHE substrate boundary wrapper ----"
end = "# ---- End Slice 13A SHE substrate boundary wrapper ----"
block = r'''
# ---- Slice 13A SHE substrate boundary wrapper ----
# The original compile workflow remains the authoritative smoke compiler for now.
# Slice 13A wraps it to attach SHE substrate-gate metadata when source artifact
# manifests are available, without changing EFG/PIRS/HSIC behavior.
try:
    _slice13a_original_run_compile = run_compile
except NameError:  # pragma: no cover
    _slice13a_original_run_compile = None


def run_compile(*, intent_path, run_dir=None, data_root="data", source_manifest=None, require_materialized_external=False):
    if _slice13a_original_run_compile is None:  # pragma: no cover
        raise RuntimeError("original run_compile is unavailable")
    try:
        result = _slice13a_original_run_compile(
            intent_path=intent_path,
            run_dir=run_dir,
            data_root=data_root,
            source_manifest=source_manifest,
            require_materialized_external=require_materialized_external,
        )
    except TypeError as exc:
        # Preserve compatibility with pre-12B run_compile signatures during local repair/rebase workflows.
        if "unexpected keyword" not in str(exc):
            raise
        result = _slice13a_original_run_compile(intent_path=intent_path, run_dir=run_dir, data_root=data_root)

    try:
        from pegasus.output.validate import validate_output_bundle as _validate_output_bundle
        from pegasus.workflows.construct.build_substrate import run_attach_substrate_to_run as _run_attach_substrate_to_run

        run_path = result.get("run_dir") if isinstance(result, dict) else None
        if run_path is not None and source_manifest is not None:
            substrate_summary = _run_attach_substrate_to_run(run_dir=run_path, source_manifest=source_manifest)
            result["substrate_gate"] = substrate_summary
            result["validation"] = _validate_output_bundle(run_dir=str(run_path))
        elif isinstance(result, dict):
            result.setdefault("substrate_gate", {
                "status": "not_evaluated",
                "reason": "no_source_manifest_supplied_to_compile",
            })
    except Exception as exc:
        if isinstance(result, dict):
            result["substrate_gate"] = {
                "status": "failed_nonfatal",
                "reason": str(exc),
            }
    return result
# ---- End Slice 13A SHE substrate boundary wrapper ----
'''
compile_path.write_text(replace_marker_block(compile_text, start, end, block), encoding="utf-8", newline="\n")

# Patch CLI with SHE commands, preserving previous groups.
cli_path = ROOT / "src/pegasus/cli.py"
cli_text = cli_path.read_text(encoding="utf-8")
cli_block_start = "# ---- Slice 13A SHE substrate CLI ----"
cli_block_end = "# ---- End Slice 13A SHE substrate CLI ----"
cli_block = r'''
# ---- Slice 13A SHE substrate CLI ----
she_app = typer.Typer(help="Substrate Harmonization Engine inspection commands.")
app.add_typer(she_app, name="she")


@she_app.command("build-substrate")
def she_build_substrate(
    artifact: list[Path] = typer.Option([], "--artifact", help="Processed source artifact path; repeatable."),
    source_system: str = typer.Option("UNKNOWN", "--source-system"),
    provenance_mode: str = typer.Option("fixture", "--provenance-mode"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    from pegasus.workflows.construct.build_substrate import run_build_substrate_from_artifacts

    artifacts = [
        {
            "path": str(path),
            "source_system": source_system,
            "artifact_role": "processed_events",
            "provenance_mode": provenance_mode,
        }
        for path in (artifact or [])
    ]
    result = run_build_substrate_from_artifacts(artifacts=artifacts, output=output)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@she_app.command("build-substrate-manifest")
def she_build_substrate_manifest(
    source_manifest: Path = typer.Option(..., "--source-manifest"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    from pegasus.workflows.construct.build_substrate import run_build_substrate_from_source_manifest

    result = run_build_substrate_from_source_manifest(source_manifest=source_manifest, output=output)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@she_app.command("substrate-summary")
def she_substrate_summary(
    manifest: Path = typer.Option(..., "--manifest"),
) -> None:
    from pegasus.workflows.construct.build_substrate import run_substrate_summary

    typer.echo(json.dumps(run_substrate_summary(manifest=manifest), indent=2, sort_keys=True))
# ---- End Slice 13A SHE substrate CLI ----
'''
cli_path.write_text(replace_marker_block(cli_text, cli_block_start, cli_block_end, cli_block), encoding="utf-8", newline="\n")

print("Slice 13A updater applied: SHE substrate boundary, zero-variance gate, workflow, CLI, tests, and audit created.")
