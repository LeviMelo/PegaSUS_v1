from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pegasus.pirs.schemas import ModelInput, ModelOutput


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def fit_parametric_model(input_model: ModelInput, *, output_dir: str | Path) -> ModelOutput:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design = pl.read_parquet(input_model.design_matrix_path)
    outcome = pl.read_parquet(input_model.outcome_vector_path)
    outcome_col = input_model.outcome_field_id
    joined = design.join(outcome, on="support_id", how="inner") if outcome_col not in design.columns else design
    y = [_safe_float(v) for v in joined[outcome_col].to_list()]
    exposure_values = [max(_safe_float(v), 1.0) for v in joined[input_model.offset_field_id].to_list()] if input_model.offset_field_id else [1.0 for _ in y]
    total_y = sum(y)
    total_exposure = sum(exposure_values) or 1.0
    intercept_rate = total_y / total_exposure
    fitted = [max(intercept_rate * e, 1e-9) for e in exposure_values]
    residuals = [obs - fit for obs, fit in zip(y, fitted)]
    math = __import__("math")
    coef_rows = [{"term": "intercept_log_rate", "estimate": math.log(max(intercept_rate, 1e-12))}]
    for covariate in input_model.covariate_field_ids:
        values = [_safe_float(v) for v in joined[covariate].to_list()]
        if len(set(values)) <= 1:
            coef_rows.append({"term": covariate, "estimate": None})
        else:
            mean_x = sum(values) / len(values)
            high_y = sum(obs for obs, x in zip(y, values) if x >= mean_x) + 0.5
            low_y = sum(obs for obs, x in zip(y, values) if x < mean_x) + 0.5
            coef_rows.append({"term": covariate, "estimate": math.log(high_y / low_y)})
    coefficients_path = output_dir / "pirs_coefficients.parquet"
    fitted_path = output_dir / "pirs_fitted_values.parquet"
    residual_path = output_dir / "pirs_residual_values.parquet"
    pl.DataFrame(coef_rows).write_parquet(coefficients_path)
    pl.DataFrame({"support_id": joined["support_id"].to_list(), "fitted": fitted}).write_parquet(fitted_path)
    pl.DataFrame({"support_id": joined["support_id"].to_list(), "residual": residuals}).write_parquet(residual_path)
    diagnostics = {
        "n_obs": len(y),
        "family": input_model.family,
        "offset_field_id": input_model.offset_field_id,
        "exposure_offset_source": input_model.offset_field_id,
        "residual_mode": input_model.residual_mode,
        "mean_abs_residual": sum(abs(r) for r in residuals) / max(len(residuals), 1),
        "residual_values_path": str(residual_path),
    }
    return ModelOutput(
        model_id="pirs_model_poisson_all_deaths",
        status="fitted",
        family=input_model.family,
        coefficients_path=str(coefficients_path),
        diagnostics=diagnostics,
        fitted_values_path=str(fitted_path),
        residual_field_id="pirs_residual_all_deaths",
        warnings=("pirs_fixture_model_not_for_inference",),
    )
