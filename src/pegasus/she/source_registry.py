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
