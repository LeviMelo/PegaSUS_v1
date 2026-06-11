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

    def as_manifest(self) -> dict[str, Any]:
        return {
            "certification_id": self.certification_id,
            "field_id": self.field_id,
            "status": self.status,
            "denominator_uncertainty_declared": self.denominator_uncertainty_declared,
            "survey_uncertainty_declared": self.survey_uncertainty_declared,
            "calibration_dataset_hash": self.calibration_dataset_hash,
            "warnings": list(self.warnings),
        }


def blocked_certification_row(*, field_id: str) -> STDFMCertificationRow:
    return STDFMCertificationRow(
        certification_id=f"cert_pending::{field_id}",
        field_id=field_id,
        status="blocked_solver_pending",
        denominator_uncertainty_declared=False,
        survey_uncertainty_declared=False,
        calibration_dataset_hash=None,
        warnings=("stdfm_certification_required", "blocked_solver_pending"),
    )


def assert_verified_promotion_allowed(row: STDFMCertificationRow) -> None:
    if row.status != "verified":
        raise STDFMCertificationError("ST-DFM verified promotion requires certification status='verified'.")
    if not (row.denominator_uncertainty_declared or row.survey_uncertainty_declared):
        raise STDFMCertificationError(
            "ST-DFM proportion/latent field promotion requires denominator or survey uncertainty."
        )
