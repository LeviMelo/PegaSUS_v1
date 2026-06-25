from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SIDRARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    variables: list[str]
    periods: list[str]
    locality_level: str
    localities: list[str]
    classifications: dict[str, list[str]] = Field(default_factory=dict)


class SIDRATableMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    name: str
    variables: list[str]
    periods: list[str]
    locality_levels: list[str]
    localities_by_level: dict[str, list[str]]
    classifications: dict[str, list[str]] = Field(default_factory=dict)
    units_by_variable: dict[str, str | None] = Field(default_factory=dict)


class SIDRAMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tables: dict[str, SIDRATableMetadata]


class SIDRAChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    table_id: str
    variables: list[str]
    periods: list[str]
    locality_level: str
    localities: list[str]
    classifications: dict[str, list[str]]
    estimated_cells: int
    request_url: str
    request_params: dict


class SIDRAFactRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    variable_id: str
    period: str

    locality_level: str
    locality_id: str

    classification_tuple: tuple[tuple[str, str], ...]
    category_tuple: tuple[tuple[str, str], ...]

    value_raw: str | None
    value_numeric: float | None
    value_status: Literal[
        "numeric",
        "blank",
        "dash_zero_or_nil",
        "not_available",
        "suppressed_or_unidentified",
        "non_numeric_symbol",
        "header_row",
        "parse_error",
    ]

    unit: str | None
    request_hash: str
    metadata_hash: str
    fetched_at: str


class DenominatorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "official_sidra_anchor",
        "independent_population_tensor",
        "sim_informed_population_tensor",
        "blocked_missing",
    ]
    source: str
    provenance: list[str]
    state: str
    dashboard_safe: bool | Literal["warning"]
    allowed_for_rates: bool
    warnings: list[str]


class ProjectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["projected", "blocked", "failed"]
    projection_matrix_id: str | None
    warnings: list[str]
    reason: str | None = None


class BoundedPushforwardResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["bounded", "not_required", "blocked", "failed"]
    axes_kept: list[str]
    axes_dropped: list[str]
    warnings: list[str]
    reason: str | None = None
