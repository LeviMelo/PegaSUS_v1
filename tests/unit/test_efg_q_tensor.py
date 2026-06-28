from __future__ import annotations

from pegasus.core.enums import FieldState, MaterializationState
from pegasus.efg.lineage import make_lineage
from pegasus.efg.node import make_field_node
from pegasus.efg.q_tensor import compute_q_state


def test_compute_q_state_populates_diagnostics_and_moran_corrects_n_eff() -> None:
    field = make_field_node(
        name="q_fixture",
        kind="extensive_measure",
        carrier="Deaths",
        unit="counts",
        support={"n_events": 6.0, "missingness": 0.0, "denom_fragility": 0.0},
        axes={"geography": "municipality", "time": "year"},
        aggregation="additive",
        role=["fixture"],
        source=["fixture"],
        operator="fixture",
        provenance=["official"],
        state=FieldState.fragile,
        warnings=[],
        lineage=make_lineage(parent_ids=[], operator_type="fixture", operator_params={}),
        materialization_state=MaterializationState.metadata_only,
        dashboard_safe=False,
    )

    q = compute_q_state(field=field, tensor=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert q.cv is not None and q.cv > 0
    assert q.moran_i is not None and q.moran_i > 0
    assert q.temporal_roughness is not None and q.temporal_roughness > 0
    assert q.spatial_entropy is not None and q.spatial_entropy > 0
    assert q.n_eff is not None and q.n_eff < 6.0

