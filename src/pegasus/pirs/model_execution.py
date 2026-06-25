
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
from pegasus.output.bundle_manager import OutputBundleManager


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


def _fit_poisson_irls(
    *,
    y: np.ndarray,
    X: np.ndarray,
    offset_vector: np.ndarray,
    term_names: Sequence[str],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, Any]:
    beta = np.zeros(X.shape[1], dtype=float)
    if len(y):
        mean_rate = np.mean(y / np.exp(np.clip(offset_vector, -30.0, 30.0)))
        beta[0] = math.log(max(float(mean_rate), 1e-12))
    converged = False
    iterations = 0
    ridge = np.eye(X.shape[1], dtype=float) * 1e-8
    for iteration in range(1, max_iter + 1):
        eta = np.clip(offset_vector + X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        weights = np.maximum(mu, 1e-9)
        z = X @ beta + (y - mu) / weights
        xw = X * np.sqrt(weights)[:, None]
        zw = z * np.sqrt(weights)
        lhs = xw.T @ xw + ridge
        rhs = xw.T @ zw
        try:
            next_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            next_beta = np.linalg.pinv(lhs) @ rhs
        iterations = iteration
        if float(np.max(np.abs(next_beta - beta))) < tol:
            beta = next_beta
            converged = True
            break
        beta = next_beta
    fitted = np.exp(np.clip(offset_vector + X @ beta, -30.0, 30.0))
    # True Poisson Deviance Residuals computation
    d_i = np.zeros_like(y)
    for i in range(len(y)):
        if y[i] > 0:
            d_i[i] = 2.0 * (y[i] * math.log(y[i] / max(fitted[i], 1e-12)) - (y[i] - fitted[i]))
        else:
            d_i[i] = 2.0 * fitted[i]
    
    standardized = np.sign(y - fitted) * np.sqrt(np.maximum(d_i, 0.0))
    residual = y - fitted
    return {
        "terms": list(term_names),
        "coefficients": [float(v) for v in beta.tolist()],
        "observed": [float(v) for v in y.tolist()],
        "fitted": [float(v) for v in fitted.tolist()],
        "residual": [float(v) for v in residual.tolist()],
        "standardized_residual": [float(v) for v in standardized.tolist()],
        "rmse": float(math.sqrt(float(np.mean(np.square(residual))))) if len(residual) else None,
        "mean_residual": float(np.mean(residual)) if len(residual) else None,
        "sigma": float(np.std(standardized, ddof=1)) if len(standardized) > 1 else 0.0,
        "solver": "poisson_irls",
        "irls_iterations": iterations,
        "irls_converged": converged,
    }


def _fit_least_squares(*, rows: Sequence[Mapping[str, Any]], response_column: str, covariate_columns: Sequence[str], offset_column: str | None, family: str) -> dict[str, Any]:
    y = np.asarray([float(row[response_column]) for row in rows], dtype=float)
    x_cols = [np.ones(len(rows), dtype=float)]
    term_names = ["intercept"]
    for col in covariate_columns:
        raw_vals = [row.get(col) for row in rows]
        valid_vals = [float(v) for v in raw_vals if v is not None and str(v).replace('.', '', 1).isdigit() and str(v).lower() not in {"nan", "inf", "-inf"}]
        median_val = float(np.median(valid_vals)) if valid_vals else 0.0
        
        imputed = []
        indicators = []
        for v in raw_vals:
            if v is None or not str(v).replace('.', '', 1).isdigit() or str(v).lower() in {"nan", "inf", "-inf"}:
                imputed.append(median_val)
                indicators.append(1.0)
            else:
                imputed.append(float(v))
                indicators.append(0.0)
                
        x_cols.append(np.asarray(imputed, dtype=float))
        if sum(indicators) > 0:
            x_cols.append(np.asarray(indicators, dtype=float))
            term_names.extend([col, f"{col}_is_missing"])
        else:
            term_names.append(col)
    X = np.column_stack(x_cols)
    offset_vector = np.zeros(len(rows), dtype=float)
    transformed_y = y.copy()
    if family == "poisson_count_with_log_offset":
        if offset_column is not None:
            raw_offset = np.asarray([max(float(row[offset_column]), 1e-12) for row in rows], dtype=float)
            offset_vector = np.log(raw_offset)
        if np.any(y < 0):
            raise PIRSModelExecutionError("Poisson PIRS response contains negative counts")
        return _fit_poisson_irls(y=y, X=X, offset_vector=offset_vector, term_names=term_names)
    if family in {"negative_binomial", "gamma", "hurdle", "zero_inflated", "dirichlet"}:
        raise NotImplementedError(f"MSD 6.2 required model family '{family}' is not yet implemented.")
    if family not in {"gaussian_identity", "ols"}:
        # default to OLS if not strict, but maybe add warning? We will just pass through for now, as OLS is the fallback.
        pass

    if offset_column is not None:
        raw_offset = np.asarray([max(float(row[offset_column]), 1e-12) for row in rows], dtype=float)
        x_cols.append(raw_offset)
        X = np.column_stack(x_cols)
        term_names.append(offset_column)
        
    ridge = np.eye(X.shape[1], dtype=float) * 1e-8
    try:
        beta = np.linalg.solve(X.T @ X + ridge, X.T @ transformed_y)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(X, transformed_y, rcond=None)
    linear = X @ beta
    if family == "poisson_count_with_log_offset" and offset_column is not None:
        fitted = np.exp(offset_vector + linear)
    else:
        fitted = linear
    residual = y - fitted
    sigma = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    standardized = residual / sigma if sigma > 0 else residual * 0.0
    return {
        "terms": term_names,
        "coefficients": [float(v) for v in beta.tolist()],
        "observed": [float(v) for v in y.tolist()],
        "fitted": [float(v) for v in fitted.tolist()],
        "residual": [float(v) for v in residual.tolist()],
        "standardized_residual": [float(v) for v in standardized.tolist()],
        "rmse": float(math.sqrt(float(np.mean(np.square(residual))))) if len(residual) else None,
        "mean_residual": float(np.mean(residual)) if len(residual) else None,
        "sigma": sigma,
        "solver": "ordinary_least_squares",
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
    fit = _fit_least_squares(rows=rows, response_column=response_column, covariate_columns=covariate_columns, offset_column=offset_column, family=family)
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
        "row_count": len(rows),
        "coefficient_count": len(coefficients),
        "coefficients_path": str(coefficients_path),
        "fitted_values_path": str(fitted_values_path),
        "residual_values_path": str(residual_values_path),
        "diagnostics": {
            "rmse": fit["rmse"],
            "mean_residual": fit["mean_residual"],
            "sigma": fit["sigma"],
            "residual_mode": manifest.get("residual_mode"),
            "fold_scheme": manifest.get("fold_scheme"),
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
