from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class STDFMCertificationError(ValueError):
    """Raised when a ST-DFM field is promoted without certification."""


@dataclass(frozen=True)
class STDFMCertificationRow:
    certification_id: str
    field_id: str
    status: str
    denominator_uncertainty_declared: bool
    survey_uncertainty_declared: bool
    calibration_dataset_hash: str | None
    warnings: tuple[str, ...]
    mape_holdout: float | None = None
    residual_variance_ratio: float | None = None
    factor_stability: float | None = None
    observed_fraction: float | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "certification_id": self.certification_id,
            "field_id": self.field_id,
            "status": self.status,
            "denominator_uncertainty_declared": self.denominator_uncertainty_declared,
            "survey_uncertainty_declared": self.survey_uncertainty_declared,
            "calibration_dataset_hash": self.calibration_dataset_hash,
            "warnings": list(self.warnings),
            "mape_holdout": self.mape_holdout,
            "residual_variance_ratio": self.residual_variance_ratio,
            "factor_stability": self.factor_stability,
            "observed_fraction": self.observed_fraction,
        }


@dataclass(frozen=True)
class STDFMCertificationPolicy:
    verified_mape: float = 0.15
    fragile_mape: float = 0.35
    verified_variance_ratio: float = 0.25
    fragile_variance_ratio: float = 0.40
    minimum_factor_stability: float = 0.85
    minimum_observed_fraction: float = 0.5
    minimum_periods: int = 3
    minimum_spatial_localities: int = 2
    minimum_multistarts: int = 2


def certify_stdfm_metrics(
    *,
    field_id: str,
    metrics: dict[str, Any],
    observed_fraction: float,
    periods: int,
    localities: int,
    spatial_penalty: float,
    multi_starts: int,
    warnings: tuple[str, ...] = (),
    policy: STDFMCertificationPolicy | None = None,
) -> STDFMCertificationRow:
    policy = policy or STDFMCertificationPolicy()
    support_errors: list[str] = []
    if observed_fraction < policy.minimum_observed_fraction:
        support_errors.append("stdfm_observed_fraction_below_minimum")
    if periods < policy.minimum_periods:
        support_errors.append("stdfm_period_count_below_minimum")
    if spatial_penalty > 0 and localities < policy.minimum_spatial_localities:
        support_errors.append("stdfm_spatial_penalty_requires_multiple_localities")
    if support_errors:
        return STDFMCertificationRow(
            certification_id=f"cert::{field_id}",
            field_id=field_id,
            status="blocked_invalid_support",
            denominator_uncertainty_declared="proportion_denominator_unknown" not in warnings,
            survey_uncertainty_declared=False,
            calibration_dataset_hash=None,
            warnings=tuple(dict.fromkeys([*warnings, *support_errors])),
            observed_fraction=observed_fraction,
        )
    mape = metrics.get("mape_holdout")
    variance = metrics.get("residual_variance_ratio")
    stability = metrics.get("factor_stability")
    if mape is None or variance is None or stability is None:
        status = "uncertified"
    elif (
        mape <= policy.verified_mape
        and variance <= policy.verified_variance_ratio
        and stability >= policy.minimum_factor_stability
        and multi_starts >= policy.minimum_multistarts
        and "proportion_denominator_unknown" not in warnings
    ):
        status = "verified"
    elif mape <= policy.fragile_mape and variance <= policy.fragile_variance_ratio:
        status = "fragile"
    else:
        status = "failed_certification"
    return STDFMCertificationRow(
        certification_id=f"cert::{field_id}",
        field_id=field_id,
        status=status,
        denominator_uncertainty_declared="proportion_denominator_unknown" not in warnings,
        survey_uncertainty_declared=False,
        calibration_dataset_hash=None,
        warnings=warnings,
        mape_holdout=mape,
        residual_variance_ratio=variance,
        factor_stability=stability,
        observed_fraction=observed_fraction,
    )


def blocked_certification_row(*, field_id: str) -> STDFMCertificationRow:
    return STDFMCertificationRow(
        certification_id=f"cert_invalid_support::{field_id}",
        field_id=field_id,
        status="blocked_invalid_support",
        denominator_uncertainty_declared=False,
        survey_uncertainty_declared=False,
        calibration_dataset_hash=None,
        warnings=("stdfm_certification_required", "stdfm_invalid_support"),
    )


def assert_verified_promotion_allowed(row: STDFMCertificationRow) -> None:
    if row.status != "verified":
        raise STDFMCertificationError("ST-DFM verified promotion requires certification status='verified'.")
    if not (row.denominator_uncertainty_declared or row.survey_uncertainty_declared):
        raise STDFMCertificationError(
            "ST-DFM proportion/latent field promotion requires denominator or survey uncertainty."
        )
