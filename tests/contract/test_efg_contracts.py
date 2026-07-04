"""T0-2 — EFG legality / Q-tensor / declaration contracts (MSD-III §II.3, X.1).

Pins the landed EFG remediations:
    * Q-tensor carries the full MSD-I §3.12 diagnostic column set.
    * n_eff is the MSD-I §3.12.3 formula — Kish effective size × spatial deflation.
    * the race-axis declaration gate fails closed (EFG-DECL-01/02).
    * the high-cardinality SIDRA axis bound is enforced before materialization.
"""

from __future__ import annotations

import pytest

from pegasus.core.enums import FieldState, MaterializationState
from pegasus.efg.declaration import OperatorSpec, evaluate_declaration_compatibility
from pegasus.efg.legality import _high_dimensional_axes_ok
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node
from pegasus.efg.q_tensor import compute_q_state
from pegasus.output.validate import REQUIRED_Q_TENSOR_COLUMNS


def _field(**overrides):
    params = dict(
        name="f",
        kind="extensive_measure",
        carrier="Deaths",
        unit="counts",
        support={},
        axes={"geography": "municipality", "time": "year"},
        aggregation="additive",
        role=["fixture"],
        source=["fixture"],
        operator="fixture",
        provenance=["official"],
        state=FieldState.fragile,
        warnings=[],
        materialization_state=MaterializationState.metadata_only,
        dashboard_safe=False,
    )
    params.update(overrides)
    lineage = make_lineage(
        parent_ids=[],
        operator_type="fixture",
        operator_params={"name": params["name"]},
    )
    return make_field_node(lineage=lineage, **params)


# --------------------------------------------------------------------------- #
# 1. Q-tensor required columns include the MSD-I §3.12 diagnostics.
# --------------------------------------------------------------------------- #
def test_q_tensor_required_columns() -> None:
    for col in ("n_eff", "cv", "moran_i", "temporal_roughness", "spatial_entropy"):
        assert col in REQUIRED_Q_TENSOR_COLUMNS, f"Q-tensor contract dropped {col!r}"


# --------------------------------------------------------------------------- #
# 2. n_eff == (Σw)²/Σw² · 1/(1+max(0,MoranI))   [MSD-I §3.12.3]
# --------------------------------------------------------------------------- #
def test_neff_moran_correction() -> None:
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    field = _field(support={"missingness": 0.0, "denom_fragility": 0.0})
    q = compute_q_state(field=field, tensor=weights)

    assert q.moran_i is not None
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    kish = (s1 * s1) / s2  # effective sample size under unequal per-cell weights
    expected = kish / (1.0 + max(0.0, q.moran_i))  # spatial-autocorrelation deflation

    assert q.n_eff == pytest.approx(expected, rel=1e-9), (
        f"n_eff must equal MSD-I §3.12.3: Kish({kish:.4f})·1/(1+max(0,I)) "
        f"= {expected:.4f}, got {q.n_eff:.4f}"
    )
    # sanity: deflation never exceeds the Kish base, never below 1
    assert 1.0 <= q.n_eff <= kish


def test_neff_rates_use_denominator_weights() -> None:
    """For rate/proportion fields, w = denominator (MSD-I §3.12.3), not numerator."""
    numerator = [1.0, 1.0, 1.0, 1.0]
    denominator = [10.0, 90.0, 5.0, 100.0]  # concentrated → smaller Kish
    field = _field(kind="intensive_density", aggregation="weighted_mean")
    q = compute_q_state(field=field, tensor=numerator, denominator=denominator)

    s1 = sum(denominator)
    s2 = sum(w * w for w in denominator)
    kish_den = (s1 * s1) / s2
    expected = kish_den / (1.0 + max(0.0, q.moran_i or 0.0))
    assert q.n_eff == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. Race-axis declaration gate fails closed.
# --------------------------------------------------------------------------- #
def test_race_axis_declaration_gate() -> None:
    rn = OperatorSpec(name="RN", role="rate")

    # race-stratified numerator with NO declared race_axis_type → fail closed
    numerator = _field(name="race_deaths", role=["race_stratified_deaths"], axes={"geography": "municipality"})
    denominator = _field(name="pop", carrier="Population", role=["denominator"], axes={"geography": "municipality"})
    result = evaluate_declaration_compatibility(numerator=numerator, denominator=denominator, operator=rn)
    assert result.ok is False
    assert "race_axis_declaration_unverifiable" in result.warnings

    # positive control: both operands declare the same race axis → permitted
    num_ok = _field(name="race_deaths_ok", role=["race_stratified_deaths"], axes={"race_axis_type": "self_declared"})
    den_ok = _field(name="pop_ok", carrier="Population", role=["denominator"], axes={"race_axis_type": "self_declared"})
    ok_result = evaluate_declaration_compatibility(numerator=num_ok, denominator=den_ok, operator=rn)
    assert ok_result.ok is True


# --------------------------------------------------------------------------- #
# 4. High-cardinality SIDRA axis product requires a bounded pushforward.
# --------------------------------------------------------------------------- #
def test_high_card_axis_bound() -> None:
    unbounded = _field(
        name="sidra_big",
        kind="context_gradient",
        carrier="Context",
        source=["SIDRA"],
        support={"estimated_cells_raw": 60000},
    )
    ok, warnings = _high_dimensional_axes_ok(unbounded)
    assert ok is False
    assert "high_dimensional_sidra_missing_bounded_pushforward" in warnings

    bounded = _field(
        name="sidra_bounded",
        kind="context_gradient",
        carrier="Context",
        source=["SIDRA"],
        support={"estimated_cells_raw": 60000, "high_dimensional_bound": {"status": "bounded"}},
    )
    ok2, warnings2 = _high_dimensional_axes_ok(bounded)
    assert ok2 is True
