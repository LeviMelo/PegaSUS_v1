from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RegistryStatus = Literal[
    "stable",
    "calibrated",
    "mutable_solver",
    "experimental",
    "deferred",
]


class RegistryHeader(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str


class BaseRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    status: RegistryStatus
    description: str
    warnings: list[str] = Field(default_factory=list)


class GenericRegistry(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[BaseRegistryEntry]


class SourceFieldEntry(BaseRegistryEntry):
    source_system: str
    raw_field: str
    canonical_field: str
    type: str
    carrier: str
    axis_or_mark: str
    required_for: list[str] = Field(default_factory=list)
    missingness_policy: str
    parser: str
    declaration_process: str | None = None


class SourceFieldsRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[SourceFieldEntry]


class CarrierEntry(BaseRegistryEntry):
    numerator_carrier: str
    denominator_carrier: str
    role: str
    output_kind: str
    legal: bool = True


class CarrierRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[CarrierEntry]


class UnitEntry(BaseRegistryEntry):
    numerator_unit: str
    denominator_unit: str
    output_unit: str
    legal_roles: list[str]


class UnitRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[UnitEntry]


class AggregationEntry(BaseRegistryEntry):
    aggregation: str
    allowed_pushforward: bool
    allowed_pullback: bool
    rate_handling: str


class AggregationRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[AggregationEntry]


class QualityPermissionEntry(BaseRegistryEntry):
    state: str
    permissions: dict[str, Any]


class QualityPermissionsRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[QualityPermissionEntry]


class RaceAxisEntry(BaseRegistryEntry):
    source_system: str
    race_axis: str
    denominator_compatible: bool
    required_bridge_to_self_declared: str | None = None


class RaceAxisRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    entries: list[RaceAxisEntry]


class ProvenanceRule(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str


class ProvenanceRegistry(BaseModel):
    schema_version: str
    registry_version: str
    created_at: str
    updated_at: str
    provenance: str
    primitive: list[str]
    rules: list[ProvenanceRule]

    @field_validator("primitive")
    @classmethod
    def primitive_nonempty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Provenance registry must define at least one primitive label.")
        return value


class RegistryManifestItem(BaseModel):
    path: str
    schema_version: str


class RegistrySetBlock(BaseModel):
    name: str
    version: str
    created_at: str
    owner: str
    dirty_allowed: bool = False


class RegistryManifest(BaseModel):
    registry_set: RegistrySetBlock
    registries: dict[str, RegistryManifestItem]