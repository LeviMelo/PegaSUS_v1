from pathlib import Path

import pytest

from pegasus.sidra.projection import load_projection_matrix, projection_metadata
from pegasus.sidra.regime import classify_sidra_context_regime
from pegasus.sidra.stitching import SIDRASegment, stitch_sidra_longitudinal_segments
from pegasus.she.high_dimensional import HighDimensionalExposureError, bound_high_dimensional_sidra_exposure, require_bounded_pushforward
from pegasus.she.stdfm.blocked import blocked_solver_pending
from pegasus.she.stdfm.certification import STDFMCertificationError, assert_verified_promotion_allowed, blocked_certification_row
from pegasus.she.stdfm.objective import stdfm_objective_pseudocode_contract
from pegasus.she.stdfm.schema import build_stdfm_input_schema
from pegasus.she.stdfm.torch_solver import solve_stdfm


def test_sidra_stitching_preserves_segment_provenance() -> None:
    result = stitch_sidra_longitudinal_segments([
        SIDRASegment.from_mapping({"concept_id": "gdp", "table_id": "t1", "variable_id": "v", "segment_id": "a", "periods": ["2018"], "unit": "R$", "classification_version": "old"}),
        SIDRASegment.from_mapping({"concept_id": "gdp", "table_id": "t2", "variable_id": "v", "segment_id": "b", "periods": ["2019"], "unit": "R$", "classification_version": "new"}),
    ])
    assert result.status == "stitched"
    assert "sidra_table_identity_not_concept_identity" in result.warnings
    assert len(result.segment_provenance) == 2


def test_fractional_projection_matrix_warns_and_preserves_rows() -> None:
    matrix = load_projection_matrix({
        "matrix_id": "m",
        "source_axis": "raw",
        "target_axis": "target",
        "entries": [
            {"source": "x", "target": "a", "weight": 0.25},
            {"source": "x", "target": "b", "weight": 0.75},
        ],
    })
    meta = projection_metadata(measure_kind="additive", has_denominator=False, matrix=matrix)
    assert meta["status"] == "projected"
    assert meta["fractional"] is True
    assert "sidra_fractional_classification_projection" in meta["warnings"]
    assert len(matrix.as_rows()) == 2


def test_high_dimensional_bounding_blocks_illegal_aggregation() -> None:
    bound = bound_high_dimensional_sidra_exposure(
        raw_axes=["municipality", "year", "sector", "occupation"],
        demanded_axes=["municipality", "year", "sector"],
        axis_cardinalities={"municipality": 1, "year": 2, "sector": 2, "occupation": 10},
        aggregation="mean",
        high_dimensional=True,
    )
    assert bound.status == "blocked"
    with pytest.raises(HighDimensionalExposureError):
        require_bounded_pushforward(bound)


def test_stdfm_gate_and_blocked_solver_contract() -> None:
    regime = classify_sidra_context_regime(
        missing_t=True,
        schema_stable=False,
        schema_mismatch=False,
        projectable=False,
        unit="index",
        anchors_bounded=True,
        concept_compatible=True,
        temporal_points=4,
        dynamics="continuous",
    )
    assert regime.regime == "bounded_interpolate"
    assert regime.stdfm_gate is True
    inp = build_stdfm_input_schema(
        field_id="f",
        concept_id="c",
        support={"years": [2018, 2019], "municipalities_ibge_cod7": ["2704302"]},
        periods=["2018", "2019"],
        localities=["2704302"],
        transform="identity",
        dynamics="continuous",
        projection_matrix_id="m",
        stitch_metadata={"status": "stitched"},
    )
    out = solve_stdfm(inp)
    assert out.status == "blocked_solver_pending"
    assert "blocked_solver_pending" in out.warnings
    assert blocked_solver_pending(field_id="x").status == "blocked_solver_pending"


def test_stdfm_certification_blocks_verified_promotion_without_certification() -> None:
    row = blocked_certification_row(field_id="f")
    with pytest.raises(STDFMCertificationError):
        assert_verified_promotion_allowed(row)
    contract = stdfm_objective_pseudocode_contract()
    assert "Y" in contract["inputs"]
    assert "blocked_solver_pending" in contract["warnings"]
