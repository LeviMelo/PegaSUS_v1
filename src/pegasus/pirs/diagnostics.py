from __future__ import annotations

from pegasus.pirs.crossfit import FoldScheme
from pegasus.pirs.schemas import PIRSDiagnostics, PIRSSelectionResult


def build_pirs_diagnostics(*, selection: PIRSSelectionResult, fold_scheme: FoldScheme, exposure_offset_source: str | None) -> PIRSDiagnostics:
    zero_var = sum(1 for row in selection.rejected if row.get("reason") == "zero_variance_field_excluded_from_design_matrix")
    quarantined = sum(1 for row in selection.rejected if str(row.get("q_state")) in {"illegal_excluded", "blocked"})
    warnings: list[str] = []
    if zero_var:
        warnings.append("zero_variance_fields_excluded_from_design_matrix")
    if quarantined:
        warnings.append("q_state_limited_fields_excluded_from_pirs")
    if fold_scheme.residual_mode == "in_sample":
        warnings.append("fast_budget_in_sample_residuals_not_for_standard_hsic")
    return PIRSDiagnostics(
        residual_mode=fold_scheme.residual_mode,
        fold_scheme=fold_scheme.fold_scheme,
        zero_variance_rejections=zero_var,
        quarantined_rejections=quarantined,
        offset_field_id=selection.selected_offset.field_id if selection.selected_offset else None,
        exposure_offset_source=exposure_offset_source,
        warnings=tuple(warnings),
    )
