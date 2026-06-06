from pegasus.core.enums import FieldState, MaterializationState
from pegasus.efg.declaration import OperatorSpec, evaluate_declaration_compatibility
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node


def _node(name: str, carrier: str, unit: str, race_axis: str):
    lineage = make_lineage(
        parent_ids=[],
        operator_type="fixture",
        operator_params={"name": name},
    )
    return make_field_node(
        name=name,
        kind="extensive_measure",
        carrier=carrier,
        unit=unit,
        support={},
        axes={"race_axis_type": race_axis},
        aggregation="additive",
        role=["fixture"],
        source=["fixture"],
        operator="fixture",
        provenance=["official"],
        state=FieldState.fragile,
        warnings=[],
        lineage=lineage,
        materialization_state=MaterializationState.metadata_only,
        dashboard_safe=False,
    )


def test_race_axis_mismatch_fails_without_bridge():
    numerator = _node("sim_admin", "Deaths", "counts", "administrative_death_declaration")
    denominator = _node("ibge_self", "Population", "person_years", "self_declared")

    result = evaluate_declaration_compatibility(
        numerator=numerator,
        denominator=denominator,
        operator=OperatorSpec(name="RN", role="mortality_rate"),
    )

    assert not result.ok
    assert "declaration" in result.failed_terms
    assert "Bridge_R not applied" in (result.reason or "")


def test_race_axis_mismatch_passes_after_bridge_provenance():
    numerator = _node("sim_bridge", "Deaths", "counts", "administrative_death_declaration").model_copy(
        update={"provenance": ["official", "bayesian_race_axis_bridge"]}
    )
    denominator = _node("ibge_self", "Population", "person_years", "self_declared")

    result = evaluate_declaration_compatibility(
        numerator=numerator,
        denominator=denominator,
        operator=OperatorSpec(name="RN", role="mortality_rate"),
    )

    assert result.ok
