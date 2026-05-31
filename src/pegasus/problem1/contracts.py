from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from pegasus.core.hashing import sha256_json


FieldKind = Literal[
    "extensive_measure",
    "intensive_density",
    "marked_functional",
    "context_gradient",
    "bridge_divergence",
    "bridge_module",
    "observer_proxy",
    "latent_context",
]

MaterializationState = Literal[
    "unmaterialized",
    "metadata_only",
    "planned",
    "materialized",
    "cached",
    "blocked",
    "failed",
    "quarantined",
]

FieldState = Literal[
    "verified",
    "fragile",
    "forced_fragile",
    "quarantined_descriptive",
    "quarantined_nochildren",
    "illegal_excluded",
    "blocked_solver_pending",
]


class Lineage(BaseModel):
    parent_ids: list[str] = Field(default_factory=list)
    operator_type: str
    operator_params: dict[str, Any] = Field(default_factory=dict)
    registry_versions: dict[str, str] = Field(default_factory=dict)

    def stable_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class FieldNode(BaseModel):
    id: str
    name: str
    kind: FieldKind
    carrier: str
    unit: str
    support: dict[str, Any]
    axes: dict[str, Any]
    aggregation: str
    role: list[str]
    source: list[str]
    operator: str | None = None
    provenance: list[str] = Field(default_factory=list)
    state: FieldState
    dashboard_safe: bool | str = False
    warnings: list[str] = Field(default_factory=list)
    lineage: Lineage
    materialization_state: MaterializationState = "metadata_only"
    path: str | None = None

    @classmethod
    def from_lineage(
        cls,
        *,
        name: str,
        kind: FieldKind,
        carrier: str,
        unit: str,
        support: dict[str, Any],
        axes: dict[str, Any],
        aggregation: str,
        role: list[str],
        source: list[str],
        operator: str | None,
        provenance: list[str],
        state: FieldState,
        dashboard_safe: bool | str,
        warnings: list[str],
        lineage: Lineage,
        materialization_state: MaterializationState = "metadata_only",
        path: str | None = None,
    ) -> "FieldNode":
        field_id = lineage.stable_hash()
        return cls(
            id=field_id,
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
            state=state,
            dashboard_safe=dashboard_safe,
            warnings=warnings,
            lineage=lineage,
            materialization_state=materialization_state,
            path=path,
        )


class AlignmentResult(BaseModel):
    ok: bool
    aligned_left_id: str | None = None
    aligned_right_id: str | None = None
    operations_applied: list[str] = Field(default_factory=list)
    support_after_alignment: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class DeltaResult(BaseModel):
    legal: bool
    delta_support: int = 1
    delta_axes: int = 1
    delta_carrier: int = 1
    delta_unit: int = 1
    delta_aggregation: int = 1
    delta_provenance: int = 1
    delta_quality: int = 1
    delta_declaration: int = 1
    failed_terms: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failed_branch_id: str | None = None

    @field_validator(
        "delta_support",
        "delta_axes",
        "delta_carrier",
        "delta_unit",
        "delta_aggregation",
        "delta_provenance",
        "delta_quality",
        "delta_declaration",
    )
    @classmethod
    def delta_terms_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("Delta terms must be binary: 0 or 1.")
        return value


class OperatorResult(BaseModel):
    status: Literal["success", "blocked", "failed"]
    output_field_id: str | None = None
    materialization_state: MaterializationState
    provenance: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failed_branch_id: str | None = None


class WarningRecord(BaseModel):
    warning_id: str = Field(default_factory=lambda: f"warning_{uuid.uuid4().hex[:12]}")
    field_id: str | None = None
    source: str
    severity: Literal["info", "warning", "severe", "abort"]
    code: str
    message: str
    parent_warning_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FailedBranch(BaseModel):
    failed_branch_id: str = Field(default_factory=lambda: f"failed_{uuid.uuid4().hex[:12]}")
    attempted_operator: str
    parent_field_ids: list[str] = Field(default_factory=list)
    reason_code: str
    delta_result: DeltaResult | None = None
    recoverable: bool = False
    suggested_route: str | None = None


class QState(BaseModel):
    field_id: str
    n_events: float | None = None
    n_denom: float | None = None
    n_eff: float | None = None
    cov_S: float | None = None
    cov_T: float | None = None
    missingness: float | None = None
    zero_inflation: float | None = None
    denom_fragility: float | None = None
    cv: float | None = None
    moran_i: float | None = None
    temporal_roughness: float | None = None
    spatial_entropy: float | None = None
    provenance_risk: float
    race_bridge_cv: float | None = None
    sensitivity_width: float | None = None
    state: FieldState
    dashboard_safe: bool | str
    warnings: list[str] = Field(default_factory=list)