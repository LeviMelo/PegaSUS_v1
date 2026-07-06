"""LinkRecord — the typed dependency-output object (MSD-II §II.8, MII-LDO-00).

Replaces the pairwise hypothesis row (MSD §8.3) with a typed relationship record
carrying lag, direction, shape, spatial heterogeneity, latent-confounding,
stability, and certification. The LDO emits these into the ``Hypotheses`` key of
the 17-key bundle (the validator is extended additively, not broken).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EdgeType = Literal[
    "contemporaneous", "lagged_directed", "latent_shared", "nonlinear_residual",
    "mechanical_overlap",   # concept-variables sharing codes → mechanically correlated, never a discovery (§5.3)
]


@dataclass(frozen=True)
class LinkRecord:
    source_var: str
    target_var: str
    edge_type: EdgeType
    lag_k: int = 0
    weight: float = 0.0
    partial_correlation: float | None = None
    response_curve_ref: str | None = None      # ref to the distributed-lag profile {S^{(k,0)}_ij}
    spatial_field_ref: str | None = None        # ref to the link's spatial heterogeneity field
    stability: float | None = None              # stability-selection frequency
    uncertainty: float | None = None
    confounding_factor_refs: tuple[str, ...] = ()
    null_strategy: str | None = None
    fdr_method: str | None = None
    # Disease-axis provenance (§II.8 / §III.7): the code system the variables live in,
    # the SIM topology role (underlying_cause vs mention), the projection status of the
    # disease concepts, and the Jaccard overlap of the two variables' code sets.
    code_system: str | None = None
    topology_role: str | None = None
    projection_status: str | None = None
    overlap_jaccard: float | None = None
    certification_status: str | None = None
    warnings: tuple[str, ...] = ()

    def as_row(self) -> dict[str, Any]:
        """Flatten to a bundle-writable row (Hypotheses key)."""
        return {
            "source_var": self.source_var,
            "target_var": self.target_var,
            "edge_type": self.edge_type,
            "lag_k": self.lag_k,
            "weight": self.weight,
            "partial_correlation": self.partial_correlation,
            "response_curve_ref": self.response_curve_ref,
            "spatial_field_ref": self.spatial_field_ref,
            "stability": self.stability,
            "uncertainty": self.uncertainty,
            "confounding_factor_refs": list(self.confounding_factor_refs),
            "null_strategy": self.null_strategy,
            "fdr_method": self.fdr_method,
            "code_system": self.code_system,
            "topology_role": self.topology_role,
            "projection_status": self.projection_status,
            "overlap_jaccard": self.overlap_jaccard,
            "certification_status": self.certification_status,
            "warnings": list(self.warnings),
        }


LINK_RECORD_COLUMNS: tuple[str, ...] = (
    "source_var",
    "target_var",
    "edge_type",
    "lag_k",
    "weight",
    "partial_correlation",
    "response_curve_ref",
    "spatial_field_ref",
    "stability",
    "uncertainty",
    "confounding_factor_refs",
    "null_strategy",
    "fdr_method",
    "code_system",
    "topology_role",
    "projection_status",
    "overlap_jaccard",
    "certification_status",
    "warnings",
)


__all__ = ["LinkRecord", "EdgeType", "LINK_RECORD_COLUMNS"]
