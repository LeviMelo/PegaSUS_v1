from pegasus.sidra.category_maps import bounded_pushforward_scaffold
from pegasus.sidra.projection import project_classification_to_axis
from pegasus.she.population.schema import blocked_missing_population_contract, official_sidra_anchor_contract


def test_projection_blocks_ratio_without_denominator_recovery():
    result = project_classification_to_axis(
        measure_kind="proportion",
        has_denominator=False,
        projection_matrix_id="race_projection_v1",
    )
    assert result.status == "blocked"
    assert "ratio_projection_requires_denominator_recovery" in result.warnings


def test_projection_allows_additive_with_matrix():
    result = project_classification_to_axis(
        measure_kind="count",
        has_denominator=False,
        projection_matrix_id="population_axis_projection_v1",
    )
    assert result.status == "projected"


def test_bounded_pushforward_blocks_nonaggregable_high_dimensional_field():
    result = bounded_pushforward_scaffold(
        raw_axes=["age", "sex", "race"],
        demanded_axes=["sex"],
        aggregation="non_aggregable",
        high_dimensional=True,
    )
    assert result.status == "blocked"


def test_bounded_pushforward_additive_field():
    result = bounded_pushforward_scaffold(
        raw_axes=["age", "sex", "race"],
        demanded_axes=["sex"],
        aggregation="additive",
        high_dimensional=True,
    )
    assert result.status == "bounded"
    assert result.axes_dropped == ["age", "race"]


def test_population_denominator_contracts():
    official = official_sidra_anchor_contract()
    assert official.mode == "official_sidra_anchor"
    assert official.allowed_for_rates is True

    blocked = blocked_missing_population_contract(reason="missing_population")
    assert blocked.mode == "blocked_missing"
    assert blocked.allowed_for_rates is False
