from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


FieldPermissionState = Literal[
    "verified",
    "warning",
    "fragile",
    "quarantined_descriptive",
    "illegal_excluded",
    "blocked",
]

ResidualMode = Literal[
    "in_sample",
    "cross_fitted",
    "parametric_bootstrap",
    "posterior_predictive",
]

ModelFamily = Literal[
    "poisson_count_with_log_offset",
    "binomial_proportion",
    "gaussian_identity",
    "sih_gamma_cost_component",
]


@dataclass(frozen=True)
class FieldCandidate:
    field_id: str
    role: Literal["outcome", "covariate", "offset"]
    utility: float
    q_state: FieldPermissionState
    carrier: str
    unit: str
    support: dict[str, Any]
    variance: float | None = None
    warnings: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    @property
    def zero_variance(self) -> bool:
        return self.variance is not None and abs(float(self.variance)) <= 1e-12

    @property
    def model_eligible(self) -> bool:
        return self.q_state not in {"illegal_excluded", "blocked"} and not self.zero_variance


@dataclass(frozen=True)
class PIRSSelectionResult:
    selected_outcome: FieldCandidate | None
    selected_covariates: tuple[FieldCandidate, ...]
    selected_offset: FieldCandidate | None
    rejected: tuple[dict[str, Any], ...]
    budget: Literal["fast", "standard", "deep"]
    top_k: int


@dataclass(frozen=True)
class ModelInput:
    outcome_field_id: str
    covariate_field_ids: tuple[str, ...]
    offset_field_id: str | None
    family: ModelFamily
    support_index_path: str
    design_matrix_path: str
    outcome_vector_path: str
    q_state_filter: dict[str, Any]
    registry_versions: dict[str, str]
    residual_mode: ResidualMode


@dataclass(frozen=True)
class ModelOutput:
    model_id: str
    status: Literal["fitted", "failed", "skipped"]
    family: ModelFamily
    coefficients_path: str | None
    diagnostics: dict[str, Any]
    fitted_values_path: str | None
    residual_field_id: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResidualField:
    field_id: str
    parent_model_id: str
    residual_type: Literal["deviance", "pearson", "randomized_quantile", "standardized", "ilr"]
    support: dict[str, Any]
    provenance: tuple[str, ...] = ("model_derived",)
    state: FieldPermissionState = "warning"
    dashboard_safe: str = "False"
    warnings: tuple[str, ...] = ("model_derived_residual_not_raw_epidemiological_variable",)


@dataclass(frozen=True)
class PIRSDiagnostics:
    residual_mode: ResidualMode
    fold_scheme: str
    zero_variance_rejections: int
    quarantined_rejections: int
    offset_field_id: str | None
    exposure_offset_source: str | None
    warnings: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "residual_mode": self.residual_mode,
            "fold_scheme": self.fold_scheme,
            "zero_variance_rejections": self.zero_variance_rejections,
            "quarantined_rejections": self.quarantined_rejections,
            "offset_field_id": self.offset_field_id,
            "exposure_offset_source": self.exposure_offset_source,
            "warnings": list(self.warnings),
        }
