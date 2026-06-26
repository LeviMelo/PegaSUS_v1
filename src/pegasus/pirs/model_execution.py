
"""PIRS model execution and residual materialization boundary.

Slice 17B consumes the Slice 17A design matrix artifact, fits a deterministic
local residualization model, writes coefficients/fitted/residual artifacts,
and registers the model/residual relation in the 17-key output bundle. HSIC
remains blocked; compile and output validation semantics are not modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pegasus.core.io_utils import _compact, _hash_payload, _load_json, _safe_id, _write_json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pegasus.compute.glm import GLMError, crossfit_residuals, fit_glm, select_count_family

# Count families eligible for §6.2 data-aware routing and log-exposure offsets.
_COUNT_FAMILIES = {"poisson_count_with_log_offset", "negative_binomial", "quasi_poisson", "hurdle_poisson", "hurdle_nb"}
_LOG_OFFSET_FAMILIES = _COUNT_FAMILIES
from pegasus.output.bundle_manager import OutputBundleManager
from pegasus.pirs.design_matrix import _read_rows, _write_rows


DEFAULT_MATRIX_MANIFEST = Path("Tables") / "pirs_design_matrix_manifest.json"
DEFAULT_MODEL_MANIFEST = Path("Tables") / "pirs_model_execution_manifest.json"
DEFAULT_COEFFICIENTS = Path("Tables") / "pirs_model_coefficients.parquet"
DEFAULT_FITTED_VALUES = Path("Tables") / "pirs_fitted_values.parquet"
DEFAULT_RESIDUAL_VALUES = Path("Tables") / "pirs_residual_values.parquet"
MODEL_GATE_KEY = "pirs_model_execution_gate"
JSON_ATTACH_TARGETS: tuple[str, ...] = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)
RESIDUAL_WARNING = "model_derived_residual_not_raw_epidemiological_variable"


class PIRSModelExecutionError(RuntimeError):
    """Raised when model execution receives malformed or blocked input."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()




def _field_specs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = manifest.get("field_specs")
    return [dict(item) for item in specs] if isinstance(specs, list) else []


def _field_id_by_role(manifest: Mapping[str, Any], role: str) -> str | None:
    for spec in _field_specs(manifest):
        if spec.get("role") == role and spec.get("field_id") not in (None, ""):
            return str(spec.get("field_id"))
    return None


def _covariate_field_ids(manifest: Mapping[str, Any]) -> list[str]:
    return [str(spec.get("field_id")) for spec in _field_specs(manifest) if spec.get("role") == "covariate" and spec.get("field_id") not in (None, "")]


def _matrix_path(run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    raw = manifest.get("matrix_path")
    if raw in (None, ""):
        return run_dir / "Tables" / "pirs_design_matrix.parquet"
    path = Path(str(raw))
    if path.is_absolute() or path.exists():
        return path
    return run_dir / path


def _blocking_manifest(*, run_dir: Path, model_manifest_path: Path, design_matrix_manifest_path: Path, reasons: Sequence[str]) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "slice": "17B",
        "artifact": "pirs_model_execution_manifest",
        "status": "blocked",
        "model_fit_state": "blocked",
        "residual_state": "not_created",
        "hsic_state": "blocked",
        "run_dir": str(run_dir),
        "manifest_path": str(model_manifest_path),
        "source_design_matrix_manifest": str(design_matrix_manifest_path),
        "model_id": None,
        "residual_field_id": None,
        "row_count": 0,
        "coefficient_count": 0,
        "blocking_reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "mutated_output_bundle": False,
    }
    _write_json(model_manifest_path, payload)
    return payload


def _design_columns(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[str, list[str], str | None]:
    if not rows:
        return "response", [], None
    columns = set(rows[0].keys())
    response = "response" if "response" in columns else "outcome"
    covariates = [spec.get("column") for spec in _field_specs(manifest) if spec.get("role") == "covariate"]
    covariates = [str(col) for col in covariates if col in columns]
    if not covariates:
        covariates = sorted(col for col in columns if str(col).startswith("covariate_"))
    offset = next((str(spec.get("column")) for spec in _field_specs(manifest) if spec.get("role") == "offset" and spec.get("column") in columns), None)
    if offset is None and "offset" in columns:
        offset = "offset"
    return response, covariates, offset


_MISSING_TOKENS: frozenset[str] = frozenset({"", "nan", "inf", "-inf", "none", "null", "na"})


def _to_float(value: Any) -> float | None:
    """Robust numeric parse. Unlike the previous ``.isdigit()`` check, this does
    not misclassify negative or scientific-notation values as missing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    try:
        f = float(text)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _build_design(
    *,
    rows: Sequence[Mapping[str, Any]],
    response_column: str,
    covariate_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, float]]:
    """Build (y, X, term_names) with honest, non-silent missingness handling.

    MSD §2.3: missingness is an observer signal, never erased silently. For every
    covariate with missing cells we emit an explicit ``*_is_missing`` indicator
    column and impute the underlying with the observed mean, recording the
    missing share and a warning so the imputation is auditable, not invisible.
    """
    n = len(rows)
    y = np.asarray([_to_float(row.get(response_column)) or 0.0 for row in rows], dtype=float)
    x_cols: list[np.ndarray] = [np.ones(n, dtype=float)]
    term_names: list[str] = ["intercept"]
    warnings: list[str] = []
    missing_shares: dict[str, float] = {}
    for col in covariate_columns:
        parsed = [_to_float(row.get(col)) for row in rows]
        observed = [v for v in parsed if v is not None]
        fill = float(np.mean(observed)) if observed else 0.0
        missing_mask = [v is None for v in parsed]
        share = float(sum(missing_mask)) / n if n else 0.0
        values = np.asarray([fill if v is None else v for v in parsed], dtype=float)
        x_cols.append(values)
        term_names.append(col)
        if share > 0.0:
            x_cols.append(np.asarray([1.0 if m else 0.0 for m in missing_mask], dtype=float))
            term_names.append(f"{col}_is_missing")
            missing_shares[col] = share
            warnings.append(f"covariate_missingness_indicator_added:{col}:{share:.4f}")
    X = np.column_stack(x_cols) if x_cols else np.ones((n, 1))
    return y, X, term_names, warnings, missing_shares


def _block_index(rows: Sequence[Mapping[str, Any]]) -> list[int] | None:
    """Spatial/temporal block id per row for block-preserving cross-fitting."""
    keys = ("block_id", "spatial_block", "support_block", "municipality", "uf", "year")
    for key in keys:
        if rows and key in rows[0]:
            return [hash(str(row.get(key))) for row in rows]
    return None


def _fit_model(
    *,
    rows: Sequence[Mapping[str, Any]],
    response_column: str,
    covariate_columns: Sequence[str],
    offset_column: str | None,
    family: str,
    residual_mode: str,
) -> dict[str, Any]:
    """Fit a real GLM and compute the residuals HSIC will consume.

    For ``cross_fitted``/``parametric_bootstrap`` residual modes (standard/deep
    budgets) the residual field is genuinely out-of-fold — fitted on the
    complement of each fold — satisfying MSD §6.6.1 and the §10 hard-abort that
    forbids in-sample residuals for standard/deep HSIC.
    """
    y, X, term_names, missing_warnings, missing_shares = _build_design(
        rows=rows, response_column=response_column, covariate_columns=covariate_columns
    )
    requested_family = family
    # Data-aware family routing (MSD §6.2): refine count families to hurdle/NB by the
    # observed zero-mass and dispersion. The actually-fitted family is recorded honestly.
    if family in _COUNT_FAMILIES and y.size and np.all(y >= 0):
        family = select_count_family(y, base_family="poisson_count_with_log_offset")
        if family != requested_family:
            missing_warnings.append(f"family_refined_by_data:{requested_family}->{family}")
    log_offset = family in _LOG_OFFSET_FAMILIES
    offset_vec: np.ndarray | None = None
    if offset_column is not None and rows and offset_column in rows[0]:
        raw = np.asarray([max(_to_float(row.get(offset_column)) or 1e-12, 1e-12) for row in rows], dtype=float)
        offset_vec = np.log(raw) if log_offset else raw
    if family in _COUNT_FAMILIES and np.any(y < 0):
        raise PIRSModelExecutionError(f"count PIRS response contains negative values for family {family}")

    try:
        fit = fit_glm(
            y=y,
            X=X,
            family=family,
            offset=offset_vec if log_offset else None,
            term_names=term_names,
        )
    except GLMError as exc:
        raise PIRSModelExecutionError(f"glm_fit_failed:{family}:{exc}") from exc

    in_sample_residual = fit.primary_residual
    residual_vector = in_sample_residual
    actual_mode = "in_sample"
    crossfit_diag: dict[str, Any] = {}
    if residual_mode in {"cross_fitted", "parametric_bootstrap"}:
        oof, crossfit_diag = crossfit_residuals(
            y=y,
            X=X,
            family=family,
            offset=offset_vec if log_offset else None,
            n_folds=5,
            block_index=_block_index(rows),
        )
        # Fall back to in-sample only for rows no fold could cover; record it and
        # — critically — DO NOT keep claiming pure "cross_fitted" when some rows
        # are in-sample. The §10 hard-abort forbids standard/deep HSIC consuming
        # in-sample residuals, so a mixed vector must be labelled mixed and
        # warned, never silently passed off as fully out-of-fold.
        nan_mask = np.isnan(oof)
        if nan_mask.any():
            oof = oof.copy()
            oof[nan_mask] = in_sample_residual[nan_mask]
            backfilled = int(nan_mask.sum())
            crossfit_diag["in_sample_backfilled_rows"] = backfilled
            actual_mode = "cross_fitted_with_in_sample_backfill"
            missing_warnings.append(
                f"residual_mode_mixed_cross_fitted_and_in_sample_backfill:{backfilled}_rows"
            )
        else:
            actual_mode = "cross_fitted"
        residual_vector = oof

    residual = residual_vector
    return {
        "terms": fit.terms,
        "coefficients": [float(v) for v in fit.coefficients.tolist()],
        "observed": [float(v) for v in y.tolist()],
        "fitted": [float(v) for v in fit.fitted.tolist()],
        "residual": [float(v) for v in residual.tolist()],
        "standardized_residual": [float(v) for v in residual.tolist()],
        "in_sample_residual": [float(v) for v in in_sample_residual.tolist()],
        "rmse": float(math.sqrt(float(np.mean(np.square(residual))))) if len(residual) else None,
        "mean_residual": float(np.mean(residual)) if len(residual) else None,
        "sigma": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "solver": f"glm_irls:{family}",
        "family_requested": requested_family,
        "family_fitted": fit.family,
        "residual_type": fit.residual_type,
        "residual_mode_requested": residual_mode,
        "residual_mode_actual": actual_mode,
        "irls_iterations": fit.n_iter,
        "irls_converged": fit.converged,
        "deviance": fit.deviance,
        "dispersion": fit.dispersion,
        "crossfit": crossfit_diag,
        "missing_shares": missing_shares,
        "warnings": missing_warnings,
    }


def build_pirs_model_execution_manifest(*, run_dir: str | Path, design_matrix_manifest: str | Path | Mapping[str, Any] | None = None, output_manifest: str | Path | None = None, mutate_output_bundle: bool = True, validate: bool = True, bundle: OutputBundleManager | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    design_manifest_path = Path(design_matrix_manifest) if isinstance(design_matrix_manifest, (str, Path)) else root / DEFAULT_MATRIX_MANIFEST
    model_manifest_path = Path(output_manifest) if output_manifest is not None else root / DEFAULT_MODEL_MANIFEST
    manifest = dict(design_matrix_manifest) if isinstance(design_matrix_manifest, Mapping) else _load_json(design_manifest_path)
    reasons: list[str] = []
    if manifest.get("status") != "ready":
        reasons.append(f"design_matrix_manifest_not_ready:{manifest.get('status', 'missing')}")
    if manifest.get("matrix_written") is not True:
        reasons.append("design_matrix_not_written")
    matrix_path = _matrix_path(root, manifest)
    if not matrix_path.exists():
        reasons.append(f"design_matrix_file_missing:{matrix_path}")
    if reasons:
        return _blocking_manifest(run_dir=root, model_manifest_path=model_manifest_path, design_matrix_manifest_path=design_manifest_path, reasons=reasons)

    rows = _read_rows(matrix_path)
    if not rows:
        return _blocking_manifest(run_dir=root, model_manifest_path=model_manifest_path, design_matrix_manifest_path=design_manifest_path, reasons=["design_matrix_has_no_rows"])
    response_column, covariate_columns, offset_column = _design_columns(manifest, rows)
    if response_column not in rows[0]:
        return _blocking_manifest(run_dir=root, model_manifest_path=model_manifest_path, design_matrix_manifest_path=design_manifest_path, reasons=[f"response_column_missing:{response_column}"])
    family = str(manifest.get("family") or "gaussian_identity")
    outcome_field_id = _field_id_by_role(manifest, "outcome") or response_column
    covariate_field_ids = _covariate_field_ids(manifest)
    offset_field_id = _field_id_by_role(manifest, "offset")
    model_basis = {"outcome": outcome_field_id, "covariates": covariate_field_ids, "offset": offset_field_id, "family": family, "matrix_manifest": str(design_manifest_path), "row_count": len(rows)}
    digest = _hash_payload(model_basis)[:12]
    model_id = f"pirs_model_{_safe_id(outcome_field_id)}_{digest}"
    residual_field_id = f"pirs_residual_{_safe_id(outcome_field_id)}_{digest}"
    residual_mode = str(manifest.get("residual_mode") or "in_sample")
    fit = _fit_model(
        rows=rows,
        response_column=response_column,
        covariate_columns=covariate_columns,
        offset_column=offset_column,
        family=family,
        residual_mode=residual_mode,
    )
    coefficients = [{"model_id": model_id, "term": term, "coefficient": coef, "term_index": i, "family": family} for i, (term, coef) in enumerate(zip(fit["terms"], fit["coefficients"], strict=False))]
    fitted_rows = []
    residual_rows = []
    for i, (observed, fitted, residual, standardized) in enumerate(zip(fit["observed"], fit["fitted"], fit["residual"], fit["standardized_residual"], strict=False)):
        row_id = rows[i].get("row_id", i)
        fitted_rows.append({"model_id": model_id, "row_id": row_id, "observed": observed, "fitted": fitted, "residual": residual, "standardized_residual": standardized})
        residual_rows.append({"field_id": residual_field_id, "model_id": model_id, "row_id": row_id, "residual": residual, "standardized_residual": standardized})
    coefficients_path = root / DEFAULT_COEFFICIENTS
    fitted_values_path = root / DEFAULT_FITTED_VALUES
    residual_values_path = root / DEFAULT_RESIDUAL_VALUES
    _write_rows(coefficients_path, coefficients)
    _write_rows(fitted_values_path, fitted_rows)
    _write_rows(residual_values_path, residual_rows)

    payload = {
        "schema_version": "1.0",
        "slice": "17B",
        "artifact": "pirs_model_execution_manifest",
        "status": "fitted",
        "model_fit_state": "fitted",
        "residual_state": "materialized",
        "hsic_state": "blocked",
        "run_dir": str(root),
        "manifest_path": str(model_manifest_path),
        "source_design_matrix_manifest": str(design_manifest_path),
        "design_matrix_path": str(matrix_path),
        "model_id": model_id,
        "residual_field_id": residual_field_id,
        "outcome_field_id": outcome_field_id,
        "covariate_field_ids": covariate_field_ids,
        "offset_field_id": offset_field_id,
        "family": family,
        "family_fitted": fit.get("family_fitted", family),
        "row_count": len(rows),
        "coefficient_count": len(coefficients),
        "coefficients_path": str(coefficients_path),
        "fitted_values_path": str(fitted_values_path),
        "residual_values_path": str(residual_values_path),
        "diagnostics": {
            "rmse": fit["rmse"],
            "mean_residual": fit["mean_residual"],
            "sigma": fit["sigma"],
            "residual_mode": fit.get("residual_mode_actual"),
            "residual_mode_requested": fit.get("residual_mode_requested"),
            "residual_type": fit.get("residual_type"),
            "fold_scheme": manifest.get("fold_scheme"),
            "crossfit": fit.get("crossfit"),
            "deviance": fit.get("deviance"),
            "dispersion": fit.get("dispersion"),
            "missing_shares": fit.get("missing_shares"),
            "solver": fit.get("solver"),
            "irls_iterations": fit.get("irls_iterations"),
            "irls_converged": fit.get("irls_converged"),
        },
        "blocking_reasons": [],
        "mutated_output_bundle": False,
    }
    if mutate_output_bundle:
        if bundle is None:
            raise RuntimeError("PIRS execution requires an OutputBundleManager")
        _mutate_output_bundle_with_model_result(root, payload, residual_vector=fit["residual"], bundle=bundle)
        payload["mutated_output_bundle"] = True
        if validate and bundle is None:
            from pegasus.output.validate import validate_output_bundle

            validation = validate_output_bundle(run_dir=str(root))
            payload["output_validation"] = {"ok": validation.ok, "errors": validation.errors, "warnings": validation.warnings}
            if not validation.ok:
                payload["status"] = "invalid_output_bundle"
                payload["model_fit_state"] = "fitted_but_invalid_output_bundle"
    summary = pirs_model_execution_summary(payload, manifest_path=model_manifest_path)
    payload["summary"] = summary
    _write_json(model_manifest_path, payload)
    return payload


def _mutate_output_bundle_with_model_result(root: Path, payload: Mapping[str, Any], *, residual_vector: Sequence[float], bundle: OutputBundleManager | None = None) -> None:
    residual_field_id = str(payload["residual_field_id"])
    model_id = str(payload["model_id"])
    outcome = str(payload.get("outcome_field_id") or "")
    covariates = list(payload.get("covariate_field_ids") or [])
    support = {"row_count": payload.get("row_count"), "source_design_matrix_manifest": payload.get("source_design_matrix_manifest")}
    axes = {"support_index": "design_matrix_row_id"}
    warning_id = f"warning_{residual_field_id}"
    residual_field = {
        "field_id": residual_field_id,
        "name": f"PIRS model residual for {outcome}",
        "kind": "model_residual",
        "carrier": "model_residual",
        "unit": "residual",
        "aggregation": "non_aggregable",
        "role": _compact(["model_residual", "pirs", "diagnostic_residual"]),
        "source": _compact(["PIRS", "model_execution"]),
        "support_json": _compact(support),
        "axes_json": _compact(axes),
        "operator": "pirs_model_execution_residualization",
        "provenance": _compact(["model_derived"]),
        "state": "warning",
        "dashboard_safe": "False",
        "warnings": _compact([RESIDUAL_WARNING]),
        "lineage_hash": _hash_payload({"model_id": model_id, "residual_field_id": residual_field_id, "support": support}),
        "registry_hash": "slice17b_pirs_model_execution_v1",
        "materialization_state": "materialized",
        "path": str(root / DEFAULT_RESIDUAL_VALUES),
    }
    q_row = {
        "field_id": residual_field_id,
        "n_events": None,
        "n_denom": None,
        "n_eff": float(len(residual_vector)),
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "cv": None,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.4,
        "state": "warning",
        "dashboard_safe": "False",
        "warnings": _compact([RESIDUAL_WARNING]),
        "computed_at": _now(),
        "q_schema_version": "1.0",
        "values_json": _compact([float(v) for v in residual_vector]),
        "value_vector_json": _compact([float(v) for v in residual_vector]),
    }
    vd_row = {
        "field_id": residual_field_id,
        "display_name": residual_field["name"],
        "technical_name": residual_field_id,
        "definition": "Model-derived PIRS residual vector produced from a validated design matrix.",
        "estimand_label": "PIRS model residual",
        "source_systems": residual_field["source"],
        "carrier": residual_field["carrier"],
        "unit": residual_field["unit"],
        "support_description": residual_field["support_json"],
        "axis_description": residual_field["axes_json"],
        "provenance_description": residual_field["provenance"],
        "state": residual_field["state"],
        "dashboard_safe": residual_field["dashboard_safe"],
        "interpretation_warning": RESIDUAL_WARNING,
    }
    model_assoc = {"id": model_id, "model_id": model_id, "status": payload.get("status"), "family": payload.get("family"), "outcome_field_id": outcome, "covariate_field_id": _compact(covariates), "covariate_field_ids": _compact(covariates), "offset_field_id": payload.get("offset_field_id"), "residual_field_id": residual_field_id, "diagnostics_json": _compact(payload.get("diagnostics", {})), "created_at": _now()}
    residual_assoc = {"id": residual_field_id, "residual_association_id": residual_field_id, "residual_field_id": residual_field_id, "model_id": model_id, "parent_model_id": model_id, "outcome_field_id": outcome, "residual_type": "raw_response_residual", "status": "materialized", "created_at": _now()}
    warning = {"warning_id": warning_id, "field_id": residual_field_id, "source": "PIRS", "severity": "warning", "code": RESIDUAL_WARNING, "message": "PIRS residual is a model-derived diagnostic field, not a raw epidemiological observation.", "inherited_from": "[]", "created_at": _now()}
    if bundle is not None:
        bundle.append_table("V_fields", [residual_field])
        bundle.append_table("Q_tensor", [q_row])
        bundle.append_table("VariableDictionary", [vd_row])
        bundle.append_table("ModelAssociations", [model_assoc])
        bundle.append_table("ResidualAssociations", [residual_assoc])
        bundle.append_table("Warnings", [warning])
    else:
        raise RuntimeError("PIRS execution requires an OutputBundleManager")


def pirs_model_execution_summary(payload: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slice": "17B",
        "gate": MODEL_GATE_KEY,
        "status": payload.get("status"),
        "model_fit_state": payload.get("model_fit_state"),
        "residual_state": payload.get("residual_state"),
        "hsic_state": payload.get("hsic_state", "blocked"),
        "model_id": payload.get("model_id"),
        "residual_field_id": payload.get("residual_field_id"),
        "row_count": payload.get("row_count", 0),
        "coefficient_count": payload.get("coefficient_count", 0),
        "mutated_output_bundle": bool(payload.get("mutated_output_bundle")),
        "blocking_reason_count": len(payload.get("blocking_reasons") or []),
        "manifest_path": str(manifest_path) if manifest_path is not None else payload.get("manifest_path"),
        "output_validation": payload.get("output_validation"),
    }


def attach_pirs_model_execution_gate_to_run(*, run_dir: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(run_dir)
    for rel in JSON_ATTACH_TARGETS:
        path = root / rel
        doc = _load_json(path)
        doc[MODEL_GATE_KEY] = dict(summary)
        if rel == "ReproducibilityManifest.json":
            pirs = doc.setdefault("pirs", {})
            if isinstance(pirs, dict):
                pirs["model_execution"] = dict(summary)
            telemetry = doc.setdefault("telemetry", {})
            if isinstance(telemetry, dict):
                stage_status = telemetry.setdefault("stage_status", {})
                if isinstance(stage_status, dict):
                    stage_status["pirs_model"] = "success" if summary.get("status") == "fitted" else "blocked"
                    stage_status["pirs_hsic"] = "blocked"
        _write_json(path, doc)


def execute_pirs_model_from_design_matrix(*, run_dir: str | Path, design_matrix_manifest: str | Path | Mapping[str, Any] | None = None, output_manifest: str | Path | None = None, mutate_output_bundle: bool = True, validate: bool = True, attach: bool = True, bundle: OutputBundleManager | None = None) -> dict[str, Any]:
    payload = build_pirs_model_execution_manifest(run_dir=run_dir, design_matrix_manifest=design_matrix_manifest, output_manifest=output_manifest, mutate_output_bundle=mutate_output_bundle, validate=validate, bundle=bundle)
    summary = pirs_model_execution_summary(payload, manifest_path=payload.get("manifest_path"))
    payload["summary"] = summary
    _write_json(Path(str(payload["manifest_path"])), payload)
    if attach:
        attach_pirs_model_execution_gate_to_run(run_dir=run_dir, summary=summary)
    return {"manifest_path": payload["manifest_path"], MODEL_GATE_KEY: summary, "model_execution": payload}


def inspect_pirs_model_execution_manifest(manifest: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(manifest))
    if not payload:
        raise FileNotFoundError(f"missing or invalid PIRS model execution manifest: {manifest}")
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return pirs_model_execution_summary(payload, manifest_path=manifest)
