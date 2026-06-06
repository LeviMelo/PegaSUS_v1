from __future__ import annotations

from typing import Literal

from pegasus.core.enums import FieldState, MaterializationState
from pegasus.core.schemas import FieldNode, Lineage
from pegasus.efg.lineage import field_id_from_lineage


def make_field_node(
    *,
    name: str,
    kind: Literal[
        "extensive_measure",
        "intensive_density",
        "marked_functional",
        "context_gradient",
        "bridge_divergence",
        "bridge_module",
        "observer_proxy",
        "latent_context",
        "model_residual",
    ],
    carrier: str,
    unit: str,
    support: dict,
    axes: dict,
    aggregation: Literal[
        "additive",
        "weighted_mean",
        "statistical_functional",
        "compositional",
        "non_aggregable",
    ],
    role: list[str],
    source: list[str],
    operator: str | None,
    provenance: list[str],
    state: FieldState | str,
    warnings: list[str],
    lineage: Lineage,
    materialization_state: MaterializationState | str,
    path: str | None = None,
    dashboard_safe: bool | Literal["warning"] = False,
) -> FieldNode:
    return FieldNode(
        id=field_id_from_lineage(lineage),
        name=name,
        kind=kind,
        carrier=carrier,
        unit=unit,
        support=support,
        axes=axes,
        aggregation=aggregation,
        role=role,
        source=source,
        operator=operator,
        provenance=provenance,
        state=FieldState(state) if isinstance(state, str) else state,
        warnings=warnings,
        lineage=lineage,
        materialization_state=(
            MaterializationState(materialization_state)
            if isinstance(materialization_state, str)
            else materialization_state
        ),
        path=path,
        dashboard_safe=dashboard_safe,
    )
