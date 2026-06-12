from __future__ import annotations

from pegasus.registries.aggregation import get_aggregation
from pegasus.registries.carrier import get_carrier
from pegasus.registries.source_fields import resolve_source_field_entry, source_field_registry_summary
from pegasus.registries.unit import get_unit


def test_slice13b_source_field_registry_loads_yaml_and_core_dimensions() -> None:
    summary = source_field_registry_summary()
    assert summary["entry_count"] >= 40
    assert summary["admissible_entry_count"] > 10
    assert summary["by_source_system"]["SIM-DO"] >= 10
    assert summary["pattern_count"] >= 2

    deaths = get_carrier("Deaths")
    assert "counts" in deaths.allowed_units
    assert deaths.default_aggregation == "additive"

    counts = get_unit("counts")
    assert counts.additive is True

    additive = get_aggregation("additive")
    assert additive.allowed_for_rates_numerator is True


def test_slice13b_registry_resolves_known_sim_and_sih_fields() -> None:
    sim = resolve_source_field_entry(source_system="SIM-DO", column_name="underlying_icd_norm")
    assert sim.carrier == "Deaths"
    assert sim.unit == "counts"
    assert sim.aggregation == "additive"
    assert sim.quality_role == "diagnostic_code"
    assert sim.admissible is True
    assert "diagnostic_topology" in sim.provenance

    sih_cost = resolve_source_field_entry(source_system="SIH-RD", column_name="hospital_service_cost_real")
    assert sih_cost.carrier == "HospitalAdmissions"
    assert sih_cost.unit == "BRL"
    assert sih_cost.axes["cost_component"] == "VAL_SH"
    assert "component_specific_cost" in sih_cost.provenance


def test_slice13b_registry_patterns_preserve_cnes_vector_indices() -> None:
    qtleit = resolve_source_field_entry(source_system="CNES-ST", column_name="QTLEIT05")
    assert qtleit.carrier == "Facilities"
    assert qtleit.unit == "beds"
    assert qtleit.axes["capacity_family"] == "QTLEIT"
    assert qtleit.matched_pattern is not None
    assert "vector_indexed_capacity" in qtleit.provenance


def test_slice13b_unknown_field_is_audit_only_not_analytic() -> None:
    unknown = resolve_source_field_entry(source_system="SIM-DO", column_name="UNDECLARED_COLUMN")
    assert unknown.carrier == "AuditMetadata"
    assert unknown.admissible is False
    assert unknown.warning == "unknown_source_field_requires_registry_entry"
