from __future__ import annotations

from types import SimpleNamespace

from pegasus.efg.bridges import bridge_summary, plan_bridge_candidates
from pegasus.efg.core_seed import build_core_seed_set, classify_core_seed, core_seed_summary


def _lineage():
    return SimpleNamespace(registry_versions={"test_registry": "v1"}, source_manifest_hashes=["hash_a"])


def _field(field_id, *, carrier, unit, aggregation="additive", role=None, source=None, name=None):
    return SimpleNamespace(
        id=field_id,
        field_id=field_id,
        name=name or field_id,
        kind="extensive_measure",
        carrier=carrier,
        unit=unit,
        aggregation=aggregation,
        role=role or [],
        source=source or ["test"],
        lineage=_lineage(),
        warnings=[],
    )


def test_slice28x_core_seed_classifies_major_seed_roles():
    death = _field("sim_deaths", carrier="Deaths", unit="counts", source=["SIM"])
    population = _field("sidra_population", carrier="Population", unit="persons", source=["SIDRA"])
    diagnostic = _field("sim_CAUSABAS", carrier="Deaths", unit="ICD10", aggregation="non_aggregable", source=["SIM"])

    assert classify_core_seed(death).seed_role == "death_event_seed"
    assert classify_core_seed(population).seed_role == "population_denominator_seed"
    assert classify_core_seed(diagnostic).seed_role == "diagnostic_observer_seed"

    seed_set = build_core_seed_set([death, population, diagnostic])
    summary = core_seed_summary(seed_set)
    assert summary["seed_count"] == 3
    assert summary["role_counts"]["diagnostic_observer_seed"] == 1


def test_slice28x_bridge_planner_emits_metadata_bridge_candidates():
    fields = [
        _field("sim_deaths", carrier="Deaths", unit="counts", source=["SIM"]),
        _field("sidra_population", carrier="Population", unit="persons", source=["SIDRA"]),
        _field("sih_admissions", carrier="HospitalAdmissions", unit="counts", source=["SIH"]),
        _field("cnes_beds", carrier="Facilities", unit="beds", source=["CNES"]),
    ]
    plan = plan_bridge_candidates(fields)
    summary = bridge_summary(plan)
    assert summary["candidate_count"] >= 3
    assert summary["bridge_type_counts"]["mortality_rate_bridge"] == 1
    assert summary["bridge_type_counts"]["capacity_pressure_bridge"] == 1
