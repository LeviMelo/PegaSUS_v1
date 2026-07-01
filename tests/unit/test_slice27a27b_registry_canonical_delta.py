from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from pegasus.efg.legality import evaluate_delta
from pegasus.registries.cnes_capacity import capacity_evidence
from pegasus.registries.diagnostic_topology import diagnostic_evidence
from pegasus.registries.semantic import registry_is_scaffold_only
from pegasus.registries.sih_cost import cost_evidence
from pegasus.efg.declaration import OperatorSpec


CRITICAL_REGISTRIES = (
    "health/diagnostic_topology.yaml",
    "health/cnes_capacity_registry.yaml",
    "health/sih_cost_registry.yaml",
    "health/clinical_event_definitions.yaml",
    "fields/bridge_grammars.yaml",
    "demographic/race_axis_registry.yaml",
    "health/icd_catalog.yaml",
    "health/icd_quality_groups.yaml",
    "fields/join_affordances.yaml",
    "spatial/municipality_crosswalk_sources.yaml",
    "inference/model_registry.yaml",
    "inference/residual_registry.yaml",
)


def _enum_value(value: str):
    return SimpleNamespace(value=value)


def _field(**kwargs):
    """Return a FieldNode-like fixture while preserving raw registry identity.

    This test file intentionally does not construct a real FieldNode because the
    registry evidence helpers need the caller-supplied raw ``field_id`` values
    such as ``sim_CAUSABAS`` and ``cnes_QTLEIT``.  The fixture nevertheless must
    expose every FieldNode surface currently consumed by ``evaluate_delta``:
    state, materialization_state, provenance, support, axes, role, source,
    warnings, and id/field_id aliases.
    """

    payload = {
        "field_id": "field",
        "id": None,
        "name": "field",
        "kind": "extensive_measure",
        "carrier": "Deaths",
        "unit": "counts",
        "aggregation": "additive",
        "support": {
            "years": [2022],
            "municipalities": ["2704302"],
            "n_events": 1.0,
            "missingness": 0.0,
            "denom_fragility": 0.0,
        },
        "axes": {
            "time": "year",
            "geography": "municipality",
            "race_axis_type": None,
        },
        "role": ["test"],
        "source": ["test_fixture"],
        "operator": "raw_field",
        "provenance": ["official"],
        "state": _enum_value("verified"),
        "warnings": [],
        "lineage": SimpleNamespace(
            parent_ids=[],
            operator_type="raw_field",
            operator_params={},
            registry_versions={"registry_set": "slice27a27b_test"},
            source_manifest_hashes=["slice27a27b_test_fixture"],
            code_version="test",
        ),
        "materialization_state": _enum_value("metadata_only"),
        "path": None,
        "dashboard_safe": False,
        "role_json": "[]",
        "source_json": "[]",
        "metadata_json": "{}",
    }
    payload.update(kwargs)

    if payload["id"] is None:
        payload["id"] = payload["field_id"]

    # Allow callers to pass bare enum strings while keeping evaluate_delta's
    # field.state.value / field.materialization_state.value contract intact.
    if isinstance(payload.get("state"), str):
        payload["state"] = _enum_value(payload["state"])
    if isinstance(payload.get("materialization_state"), str):
        payload["materialization_state"] = _enum_value(payload["materialization_state"])

    return SimpleNamespace(**payload)


def test_slice27a_critical_registries_are_not_scaffold_only() -> None:
    root = Path("config/registries")
    for name in CRITICAL_REGISTRIES:
        path = root / name
        assert path.exists(), name
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Past-scaffold version pin. health/clinical_event_definitions.yaml advanced to v3.0
        # when it became the source-agnostic single source of truth for event carriers
        # and RN ratios (MSD §2.6/§3.10.4); other critical registries remain at v2.0.
        assert payload["registry_version"] in {"v2.0", "v3.0"}, name
        assert not registry_is_scaffold_only(name), name
        entries = payload.get("entries") or []
        assert len(entries) >= 1, name
        assert all("scaffold_only" not in set(entry.get("warnings") or []) for entry in entries), name


def test_slice27a_semantic_registry_helpers_classify_special_fields() -> None:
    diagnostic = _field(
        field_id="sim_CAUSABAS",
        name="CAUSABAS",
        unit="ICD10",
        aggregation="non_aggregable",
        kind="observer_proxy",
    )
    cnes = _field(
        field_id="cnes_QTLEIT",
        name="QTLEIT bed capacity",
        carrier="Facilities",
        unit="beds",
    )
    sih = _field(
        field_id="sih_VAL_SH",
        name="VAL_SH hospital service cost",
        carrier="HospitalAdmissions",
        unit="BRL",
    )
    assert diagnostic_evidence(diagnostic)["topology_role"] == "underlying_cause"
    assert capacity_evidence(cnes)["capacity_family"] == "bed_capacity"
    assert cost_evidence(sih)["cost_component"] == "hospital_service"


def test_slice27b_delta_warnings_carry_registry_evidence() -> None:
    operator = OperatorSpec(name="raw_field", role="registry_evidence", params={})
    diagnostic = _field(
        field_id="sim_CAUSABAS",
        name="CAUSABAS",
        unit="counts",
        aggregation="additive",
        kind="extensive_measure",
    )
    result = evaluate_delta(parents=[diagnostic], operator=operator)
    warnings = "\n".join(str(value) for value in result.warnings)
    assert "registry_evidence_attached" in warnings
    assert "diagnostic_topology=sim_underlying_cause_causabas" in warnings


def test_sidra_high_dimensional_field_requires_bounded_pushforward() -> None:
    operator = OperatorSpec(name="raw_field", role="registry_evidence", params={})
    sidra = _field(
        field_id="sidra_highdim_context",
        name="SIDRA high-dimensional context",
        kind="context_gradient",
        carrier="ContextCells",
        unit="raw_sidra_value",
        aggregation="additive",
        source=["SIDRA"],
        support={"high_dimensional": True, "estimated_cells_raw": 100000},
        axes={"geography": "municipality", "time": "year", "occupation": "raw_sidra"},
    )
    result = evaluate_delta(parents=[sidra], operator=operator)
    assert not result.legal
    assert "axes" in result.failed_terms
    assert "high_dimensional_sidra_missing_bounded_pushforward" in result.warnings

