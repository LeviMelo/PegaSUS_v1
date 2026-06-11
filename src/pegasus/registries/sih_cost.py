from __future__ import annotations

from dataclasses import dataclass


class SIHCostRegistryError(ValueError):
    """Raised when SIH economic estimands collapse distinct billing components."""


@dataclass(frozen=True)
class SIHCostComponent:
    raw_field: str
    component_id: str
    carrier: str
    unit: str
    label: str


COST_COMPONENTS: dict[str, SIHCostComponent] = {
    "VAL_SH": SIHCostComponent("VAL_SH", "hospital_service_cost", "HospitalCosts_SH", "reais_hospital_services", "Hospital service billing component"),
    "VAL_SP": SIHCostComponent("VAL_SP", "professional_service_cost", "HospitalCosts_SP", "reais_professional_services", "Professional service billing component"),
    "VAL_UTI": SIHCostComponent("VAL_UTI", "icu_cost", "HospitalCosts_UTI", "reais_icu_services", "ICU billing component"),
    "VAL_TOT": SIHCostComponent("VAL_TOT", "total_admission_cost", "HospitalCosts_TOT", "reais_total_billing", "Total admission billing burden"),
}


def get_cost_component(raw_field: str) -> SIHCostComponent:
    key = raw_field.upper()
    if key not in COST_COMPONENTS:
        raise SIHCostRegistryError(f"Unknown SIH cost component: {raw_field}")
    return COST_COMPONENTS[key]


def require_component_specific_cost(raw_field: str | None) -> SIHCostComponent:
    if raw_field is None or raw_field.strip().lower() in {"cost", "costs", "generic_cost", "hospital_cost", "sih_cost"}:
        raise SIHCostRegistryError("Generic SIH cost request is illegal when component-specific semantics are required.")
    return get_cost_component(raw_field)


def registry_manifest() -> dict[str, object]:
    return {"schema_version": "1.0", "registry": "sih_cost_components", "components": {k: v.__dict__ for k, v in COST_COMPONENTS.items()}}
