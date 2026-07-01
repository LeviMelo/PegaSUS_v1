from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pegasus.core.enums import FieldState, MaterializationState


class GeographySelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["municipality", "AMC", "UF", "region", "country"]
    codes: list[str]
    uf: list[str] = Field(default_factory=list)


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_year: int
    end_year: int
    start_month: int | None = None
    end_month: int | None = None


class UserIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # DataScope: which substrate exists (MSD §1.5). Orthogonal to execution_stage.
    run_profile: Literal["core_vital", "contextual", "full"] = "core_vital"
    # ExecutionStage: how far the pipeline runs (MSD-II §II.5). `validate` = SHE+EFG
    # legality only; `compile` = materialize V_fields+Q_tensor, no inference;
    # `investigate` = full pipeline incl. PIRS/LDO. Inference keys
    # (ModelAssociations/Hypotheses/…) are required-non-empty only at `investigate`.
    execution_stage: Literal["validate", "compile", "investigate"] = "investigate"
    geography: GeographySelector
    time: TimeWindow
    health_seeds: list[str] = Field(default_factory=list)
    mandatory_fields: list[str] = Field(default_factory=list)
    system_weights: dict[str, float] = Field(default_factory=dict)
    context_policy: list[str] = Field(default_factory=list)
    budget: Literal["fast", "standard", "deep"]
    geo_mode: Literal["native", "AMC", "geneallocated", "hybrid"]
    force_selectors: list[str] = Field(default_factory=list)
    exclude_systems: list[str] = Field(default_factory=list)
    # Positive year lags for delayed cross-source covariate effects (MSD §2.11),
    # e.g. [1] aligns covariate(t-1) to outcome(t) per municipality. Empty = none.
    temporal_lags: list[int] = Field(default_factory=list)
    # Spatial aggregation level for event counts (MSD §3.7): coarsen the geography
    # cell to a denser IBGE region for sparse outcomes. "municipality" (default) =
    # no aggregation.
    geography_aggregation: Literal[
        "municipality", "microregion", "immediate_region", "mesoregion", "intermediate_region"
    ] = "municipality"

    execution_scale: Literal["smoke", "state", "region", "national_blocked"]

    # decoupled: population tensor's race axis ignores DATASUS-origin admin race
    # entirely (§2.8.5/§2.8.6 race-stratified births/deaths unwired, lambda=0).
    # downstream_bridge: Bridge_R attaches as a standalone EFG field (registries.
    # race_bridge.resolve_race_bridge_plan status="planned").
    # embedded_*: the same Bridge_R prior feeds directly into the population
    # tensor's own birth/death race stratification (MSD §2.8.5/§2.8.6;
    # sidra.population_cube.build) instead of a separate field
    # (resolve_race_bridge_plan status="embedded"). The three embedded_* values
    # are currently equivalent in implementation (all use Bridge_R's point-estimate
    # posterior); fixedC/posteriorC/sensitivity distinguish how solver-side
    # uncertainty from the bridge should propagate, which is not yet implemented.
    race_tensor_mode: Literal[
        "decoupled",
        "downstream_bridge",
        "embedded_fixedC",
        "embedded_posteriorC",
        "embedded_sensitivity",
    ] = "decoupled"

    population_mode: Literal[
        "official_sidra_anchor",
        "independent_population_tensor",
        "sim_informed_population_tensor",
        "blocked_missing",
    ] = "official_sidra_anchor"


class Lineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_ids: list[str]
    operator_type: str
    operator_params: dict
    registry_versions: dict[str, str]
    source_manifest_hashes: list[str] = Field(default_factory=list)
    code_version: str


class FieldNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
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
    ]

    carrier: str
    unit: str
    support: dict
    axes: dict
    aggregation: Literal[
        "additive",
        "weighted_mean",
        "statistical_functional",
        "compositional",
        "non_aggregable",
    ]

    role: list[str]
    source: list[str]
    operator: str | None
    provenance: list[str]
    state: FieldState
    warnings: list[str]
    lineage: Lineage
    materialization_state: MaterializationState
    path: str | None = None
    dashboard_safe: bool | Literal["warning"] = False


class DATASUSRequestManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: Literal["SIM-DO", "SIH-RD", "SINASC", "CNES-ST"]
    uf: str

    year_start: int
    month_start: int | None
    year_end: int
    month_end: int | None

    information_system: str
    fetch_function: str
    process_function: str

    raw_path: str
    processed_path: str
    raw_sha256: str
    processed_sha256: str

    row_counts: dict[str, int]
    column_lists: dict[str, list[str]]

    started_at: str
    ended_at: str
    duration_seconds: float

    rscript_path: str
    r_version: str | None
    microdatasus_version: str | None
    read_dbc_version: str | None

    stdout_path: str
    stderr_path: str
    heartbeat_path: str
    exit_code: int
    status: Literal["success", "failed", "cached", "timeout", "blocked"]
    error_message: str | None
    request_hash: str


class SIDRAFact(BaseModel):
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


class QState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_id: str

    n_events: float | None
    n_denom: float | None
    n_eff: float | None
    cov_S: float | None
    cov_T: float | None
    missingness: float | None
    zero_inflation: float | None
    denom_fragility: float | None
    cv: float | None
    moran_i: float | None
    temporal_roughness: float | None
    spatial_entropy: float | None
    provenance_risk: float

    race_axis_source: str | None = None
    race_axis_target: str | None = None
    missing_race_share: float | None = None
    emission_prior_strength: float | None = None
    race_bridge_cv: float | None = None
    sensitivity_width: float | None = None
    bridge_mode: str | None = None

    state: FieldState
    dashboard_safe: bool | Literal["warning"]
    warnings: list[str]
    computed_at: str
    q_schema_version: str = "1.0"


class WarningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warning_id: str
    field_id: str | None
    source: str
    severity: Literal["info", "warning", "downgrade", "abort"]
    code: str
    message: str
    inherited_from: list[str] = Field(default_factory=list)
    created_at: str


class FailedBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failed_branch_id: str
    attempted_operator: str
    parent_field_ids: list[str]
    failure_stage: Literal[
        "alignment",
        "support",
        "axes",
        "carrier",
        "unit",
        "aggregation",
        "provenance",
        "quality",
        "declaration",
        "materialization",
        "solver",
        "output_validation",
    ]
    failed_terms: list[str]
    reason: str
    warnings: list[str]
    created_at: str


class AlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    aligned_left_id: str | None
    aligned_right_id: str | None
    operations_applied: list[str]
    support_after_alignment: dict | None
    warnings: list[str]
    failure_reason: str | None


class DeltaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legal: bool
    delta_support: int
    delta_axes: int
    delta_carrier: int
    delta_unit: int
    delta_aggregation: int
    delta_provenance: int
    delta_quality: int
    delta_declaration: int
    failed_terms: list[str]
    warnings: list[str]
    failed_branch_id: str | None


class OperatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "blocked", "failed"]
    output_field_id: str | None
    materialization_state: MaterializationState
    provenance: list[str]
    warnings: list[str]
    failed_branch_id: str | None


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
    state: FieldState
    dashboard_safe: bool | Literal["warning"]
    allowed_for_rates: bool
    warnings: list[str]
