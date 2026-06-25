
"""Run-bundle HSIC residual scan integration.

Slice 18A consumes PIRS residual artifacts emitted by Slice 17B and performs a
bounded residual-dependence scan against selected design-matrix covariates. It
writes HSIC scan artifacts and hypothesis rows but does not refit PIRS models,
re-run EFG materialization, or mutate compile semantics.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from pegasus.output.bundle_manager import OutputBundleManager

from pegasus.pirs.hsic import linear_hsic_statistic, permutation_p_value


DEFAULT_MODEL_MANIFEST = Path("Tables") / "pirs_model_execution_manifest.json"
DEFAULT_DESIGN_MANIFEST = Path("Tables") / "pirs_design_matrix_manifest.json"
DEFAULT_RESIDUAL_VALUES = Path("Tables") / "pirs_residual_values.parquet"
DEFAULT_DESIGN_MATRIX = Path("Tables") / "pirs_design_matrix.parquet"
DEFAULT_HSIC_MANIFEST = Path("Tables") / "hsic_residual_scan_manifest.json"
DEFAULT_HSIC_SCORES = Path("Tables") / "hsic_residual_scan_scores.parquet"
HSIC_GATE_KEY = "hsic_residual_scan_gate"
JSON_ATTACH_TARGETS: tuple[str, ...] = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)


class HSICResidualScanError(RuntimeError):
    """Raised when HSIC residual scanning receives malformed run artifacts."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, default=str), encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)
    return path


def _resolve_path(run_dir: Path, raw: Any, default: Path) -> Path:
    if raw in (None, ""):
        return run_dir / default
    path = Path(str(raw))
    if path.is_absolute() or path.exists():
        return path
    return run_dir / path


def _field_specs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = manifest.get("field_specs")
    return [dict(item) for item in specs] if isinstance(specs, list) else []


def _covariate_specs(design_manifest: Mapping[str, Any], matrix_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if not matrix_rows:
        return []
    columns = set(matrix_rows[0].keys())
    specs: list[dict[str, str]] = []
    for spec in _field_specs(design_manifest):
        if spec.get("role") != "covariate":
            continue
        column = spec.get("column")
        field_id = spec.get("field_id")
        if column in columns and field_id not in (None, ""):
            specs.append({"field_id": str(field_id), "column": str(column)})
    if specs:
        return specs
    for column in sorted(str(col) for col in columns if str(col).startswith("covariate_")):
        specs.append({"field_id": column, "column": column})
    return specs


def _vector_from_rows(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[float] | None:
    if not rows:
        return None
    for name in preferred:
        if name in rows[0]:
            try:
                return [float(row[name]) for row in rows]
            except (TypeError, ValueError):
                continue
    return None


def _column_vector(rows: Sequence[Mapping[str, Any]], column: str) -> list[float] | None:
    if not rows or column not in rows[0]:
        return None
    try:
        return [float(row[column]) for row in rows]
    except (TypeError, ValueError):
        return None


def _fdr_bh(p_values: Sequence[float | None]) -> list[float | None]:
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    out: list[float | None] = [None for _ in p_values]
    if m == 0:
        return out
    ranked = sorted(indexed, key=lambda item: item[1])
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(ranked), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p * m / max(rank, 1))
        out[idx] = min(1.0, max(0.0, running))
    return out


def _scan_row(*, residual_field_id: str, model_id: str | None, covariate_field_id: str, covariate_column: str, residuals: Sequence[float], covariate: Sequence[float], budget: str, permutations: int, min_support: int, seed: int) -> dict[str, Any]:
    n_eff = int(min(len(residuals), len(covariate)))
    warnings: list[str] = []
    statistic: float | None = None
    p_value: float | None = None
    mode = "exact_linear"
    state = "verified"
    if n_eff < min_support:
        mode = "disabled"
        state = "blocked"
        warnings.append("hsic_disabled_insufficient_support")
    else:
        statistic = float(linear_hsic_statistic([float(v) for v in covariate[:n_eff]], [float(v) for v in residuals[:n_eff]]))
        rng = random.Random(seed)
        null_statistics: list[float] = []
        residual_values = [float(v) for v in residuals[:n_eff]]
        covariate_values = [float(v) for v in covariate[:n_eff]]
        for _ in range(max(int(permutations), 1)):
            shuffled = residual_values[:]
            rng.shuffle(shuffled)
            null_statistics.append(linear_hsic_statistic(covariate_values, shuffled))
        p_value = float(
            permutation_p_value(
                statistic=statistic,
                permutations=permutations,
                n_eff=n_eff,
                null_statistics=null_statistics,
            )
        )
        if n_eff < 100:
            state = "fragile"
            warnings.append("hsic_descriptive_small_support")
    hypothesis_id = f"hsic__{residual_field_id}__{covariate_field_id}"
    return {
        "hypothesis_id": hypothesis_id,
        "model_id": model_id,
        "residual_field_id": residual_field_id,
        "covariate_field_id": covariate_field_id,
        "covariate_column": covariate_column,
        "hsic_mode": mode,
        "budget": budget,
        "n_eff": float(n_eff),
        "statistic": statistic,
        "p_value": p_value,
        "q_value": None,
        "state": state,
        "warnings": warnings,
        "permutations": int(permutations),
        "seed": int(seed),
        "kernel": "linear_centered",
        "created_at": _now(),
    }


def _hypothesis_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Render a scan row into the canonical Hypotheses.parquet schema.

    The output bundle Hypotheses schema is fixed and does not contain generic
    Slice 18A convenience columns such as ``method``, ``metadata_json``, or
    ``created_at``. Those details must be carried through the canonical HSIC
    columns and approximation_diagnostics_json.
    """
    warnings = row.get("warnings") if isinstance(row.get("warnings"), list) else []
    diagnostics = {
        "model_id": row.get("model_id"),
        "budget": row.get("budget"),
        "kernel": row.get("kernel"),
        "covariate_column": row.get("covariate_column"),
        "permutations": row.get("permutations"),
        "seed": row.get("seed"),
        "created_at": row.get("created_at") or _now(),
    }
    return {
        "hypothesis_id": row.get("hypothesis_id"),
        "outcome_field_id": row.get("outcome_field_id") or row.get("residual_field_id"),
        "covariate_field_id": row.get("covariate_field_id"),
        "residual_field_id": row.get("residual_field_id"),
        "statistic": row.get("statistic"),
        "p_value": row.get("p_value"),
        "q_value": row.get("q_value"),
        "hsic_mode": row.get("hsic_mode"),
        "residual_mode": row.get("residual_mode"),
        "fold_scheme": row.get("fold_scheme"),
        "bootstrap_count": row.get("bootstrap_count"),
        "residual_uncertainty": row.get("residual_uncertainty"),
        "null_strategy": row.get("null_strategy"),
        "fdr_method": row.get("fdr_method"),
        "n_eff": row.get("n_eff"),
        "state": row.get("state", "fragile"),
        "warnings": _compact(warnings),
        "approximation_diagnostics_json": _compact(diagnostics),
    }


def _warning_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for i, code in enumerate(row.get("warnings") or [], start=1):
            out.append({
                "warning_id": f"slice18a_{row.get('hypothesis_id')}_{i}",
                "field_id": row.get("residual_field_id"),
                "source": "PIRS-HSIC",
                "severity": "warning",
                "code": str(code),
                "message": str(code).replace("_", " "),
                "inherited_from": "[]",
                "created_at": _now(),
            })
    return out


def _blocking_manifest(*, run_dir: Path, manifest_path: Path, reasons: Sequence[str]) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "slice": "18A",
        "artifact": "hsic_residual_scan_manifest",
        "status": "blocked",
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "hypothesis_count": 0,
        "score_count": 0,
        "mutated_output_bundle": False,
        "blocking_reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "created_at": _now(),
    }
    _write_json(manifest_path, payload)
    return payload


def build_hsic_residual_scan_manifest(*, run_dir: str | Path, model_execution_manifest: str | Path | Mapping[str, Any] | None = None, design_matrix_manifest: str | Path | Mapping[str, Any] | None = None, output_manifest: str | Path | None = None, budget: str = "fast", permutations: int = 199, min_support: int = 3, seed: int = 20260613, mutate_output_bundle: bool = True, validate: bool = True, bundle: OutputBundleManager | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    manifest_path = Path(output_manifest) if output_manifest is not None else root / DEFAULT_HSIC_MANIFEST
    model_manifest_path = Path(model_execution_manifest) if isinstance(model_execution_manifest, (str, Path)) else root / DEFAULT_MODEL_MANIFEST
    design_manifest_path = Path(design_matrix_manifest) if isinstance(design_matrix_manifest, (str, Path)) else root / DEFAULT_DESIGN_MANIFEST
    model_manifest = dict(model_execution_manifest) if isinstance(model_execution_manifest, Mapping) else _load_json(model_manifest_path)
    design_manifest = dict(design_matrix_manifest) if isinstance(design_matrix_manifest, Mapping) else _load_json(design_manifest_path)

    reasons: list[str] = []
    if model_manifest.get("status") != "fitted":
        reasons.append(f"model_execution_not_fitted:{model_manifest.get('status', 'missing')}")
    residual_field_id = model_manifest.get("residual_field_id")
    if residual_field_id in (None, ""):
        reasons.append("residual_field_id_missing")
    residual_path = _resolve_path(root, model_manifest.get("residual_values_path"), DEFAULT_RESIDUAL_VALUES)
    if not residual_path.exists():
        reasons.append(f"residual_values_missing:{residual_path}")
    matrix_path = _resolve_path(root, design_manifest.get("matrix_path"), DEFAULT_DESIGN_MATRIX)
    if not matrix_path.exists():
        reasons.append(f"design_matrix_missing:{matrix_path}")
    if reasons:
        return _blocking_manifest(run_dir=root, manifest_path=manifest_path, reasons=reasons)

    residual_rows = _read_rows(residual_path)
    matrix_rows = _read_rows(matrix_path)
    residual_vector = _vector_from_rows(residual_rows, ("standardized_residual", "residual", "value", "values"))
    if residual_vector is None:
        return _blocking_manifest(run_dir=root, manifest_path=manifest_path, reasons=["residual_vector_not_numeric"])
    covariates = _covariate_specs(design_manifest, matrix_rows)
    if not covariates:
        return _blocking_manifest(run_dir=root, manifest_path=manifest_path, reasons=["no_covariates_available_for_hsic"])

    scan_rows: list[dict[str, Any]] = []
    for spec in covariates:
        vector = _column_vector(matrix_rows, spec["column"])
        if vector is None:
            scan_rows.append({
                "hypothesis_id": f"hsic__{residual_field_id}__{spec['field_id']}",
                "model_id": model_manifest.get("model_id"),
                "residual_field_id": str(residual_field_id),
                "covariate_field_id": spec["field_id"],
                "covariate_column": spec["column"],
                "hsic_mode": "blocked",
                "budget": budget,
                "n_eff": 0.0,
                "statistic": None,
                "p_value": None,
                "q_value": None,
                "state": "blocked",
                "warnings": ["covariate_vector_not_numeric"],
                "permutations": int(permutations),
                "seed": int(seed),
                "kernel": "linear_centered",
                "created_at": _now(),
            })
            continue
        scan_rows.append(_scan_row(residual_field_id=str(residual_field_id), model_id=model_manifest.get("model_id"), covariate_field_id=spec["field_id"], covariate_column=spec["column"], residuals=residual_vector, covariate=vector, budget=budget, permutations=permutations, min_support=min_support, seed=seed))

    q_values = _fdr_bh([row.get("p_value") for row in scan_rows])
    outcome_specs = [spec for spec in _field_specs(design_manifest) if spec.get("role") == "outcome"]
    outcome_field_id = (
        model_manifest.get("outcome_field_id")
        or (outcome_specs[0].get("field_id") if outcome_specs else None)
        or str(residual_field_id)
    )
    residual_mode = model_manifest.get("residual_mode") or design_manifest.get("residual_mode") or "in_sample"
    fold_scheme = model_manifest.get("fold_scheme") or design_manifest.get("fold_scheme") or "none_in_sample_fast_budget"
    for row, q_value in zip(scan_rows, q_values):
        row["q_value"] = q_value
        row.setdefault("outcome_field_id", outcome_field_id)
        row.setdefault("residual_mode", residual_mode)
        row.setdefault("fold_scheme", fold_scheme)
        row.setdefault("bootstrap_count", None)
        row.setdefault("residual_uncertainty", "in_sample_descriptive")
        row.setdefault("null_strategy", "permutation_linear_centered")
        row.setdefault("fdr_method", "BH")

    _write_rows(root / DEFAULT_HSIC_SCORES, scan_rows)
    payload = {
        "schema_version": "1.0",
        "slice": "18A",
        "artifact": "hsic_residual_scan_manifest",
        "status": "scanned" if scan_rows else "blocked",
        "run_dir": str(root),
        "manifest_path": str(manifest_path),
        "source_model_execution_manifest": str(model_manifest_path),
        "source_design_matrix_manifest": str(design_manifest_path),
        "residual_field_id": str(residual_field_id),
        "model_id": model_manifest.get("model_id"),
        "budget": budget,
        "permutations": int(permutations),
        "min_support": int(min_support),
        "score_path": str(root / DEFAULT_HSIC_SCORES),
        "score_count": len(scan_rows),
        "hypothesis_count": len(scan_rows),
        "blocked_count": sum(1 for row in scan_rows if row.get("state") == "blocked"),
        "fragile_count": sum(1 for row in scan_rows if row.get("state") == "fragile"),
        "verified_count": sum(1 for row in scan_rows if row.get("state") == "verified"),
        "mutated_output_bundle": bool(mutate_output_bundle),
        "blocking_reasons": [],
        "scan_rows": scan_rows,
        "created_at": _now(),
    }
    if mutate_output_bundle:
        if bundle is not None:
            bundle.append_table("Hypotheses", [_hypothesis_row(row) for row in scan_rows])
            warnings = _warning_rows(scan_rows)
            if warnings:
                bundle.append_table("Warnings", warnings)
        else:
            raise RuntimeError("HSIC execution requires an OutputBundleManager")

    if validate and bundle is None:
        try:
            from pegasus.output.validate import validate_output_bundle
            validation = validate_output_bundle(run_dir=str(root))
        except TypeError:
            from pegasus.output.validate import validate_output_bundle
            validation = validate_output_bundle(run_dir=str(root), schema_registry=None)  # type: ignore[arg-type]
        payload["output_validation"] = {"ok": bool(validation.ok), "errors": list(getattr(validation, "errors", []) or [])}
    _write_json(manifest_path, payload)
    return payload


def hsic_residual_scan_summary(payload: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slice": "18A",
        "gate": HSIC_GATE_KEY,
        "status": payload.get("status"),
        "residual_field_id": payload.get("residual_field_id"),
        "model_id": payload.get("model_id"),
        "score_count": int(payload.get("score_count") or 0),
        "hypothesis_count": int(payload.get("hypothesis_count") or 0),
        "blocked_count": int(payload.get("blocked_count") or 0),
        "fragile_count": int(payload.get("fragile_count") or 0),
        "verified_count": int(payload.get("verified_count") or 0),
        "mutated_output_bundle": bool(payload.get("mutated_output_bundle")),
        "blocking_reason_count": len(payload.get("blocking_reasons") or []),
        "manifest_path": str(manifest_path) if manifest_path is not None else payload.get("manifest_path"),
        "score_path": payload.get("score_path"),
        "output_validation": payload.get("output_validation"),
    }


def attach_hsic_residual_scan_gate_to_run(*, run_dir: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(run_dir)
    for rel in JSON_ATTACH_TARGETS:
        path = root / rel
        doc = _load_json(path)
        doc[HSIC_GATE_KEY] = dict(summary)
        if rel == "ReproducibilityManifest.json":
            pirs = doc.setdefault("pirs", {})
            if isinstance(pirs, dict):
                pirs["hsic_residual_scan"] = dict(summary)
            telemetry = doc.setdefault("telemetry", {})
            if isinstance(telemetry, dict):
                stage_status = telemetry.setdefault("stage_status", {})
                if isinstance(stage_status, dict):
                    stage_status["pirs_hsic"] = "success" if summary.get("status") == "scanned" else "blocked"
        _write_json(path, doc)


def execute_hsic_residual_scan(*, run_dir: str | Path, model_execution_manifest: str | Path | Mapping[str, Any] | None = None, design_matrix_manifest: str | Path | Mapping[str, Any] | None = None, output_manifest: str | Path | None = None, budget: str = "fast", permutations: int = 199, min_support: int = 3, seed: int = 20260613, mutate_output_bundle: bool = True, validate: bool = True, attach: bool = True, bundle: OutputBundleManager | None = None) -> dict[str, Any]:
    payload = build_hsic_residual_scan_manifest(run_dir=run_dir, model_execution_manifest=model_execution_manifest, design_matrix_manifest=design_matrix_manifest, output_manifest=output_manifest, budget=budget, permutations=permutations, min_support=min_support, seed=seed, mutate_output_bundle=mutate_output_bundle, validate=validate, bundle=bundle)
    summary = hsic_residual_scan_summary(payload, manifest_path=payload.get("manifest_path"))
    payload["summary"] = summary
    _write_json(Path(str(payload["manifest_path"])), payload)
    if attach:
        attach_hsic_residual_scan_gate_to_run(run_dir=run_dir, summary=summary)
    return {"manifest_path": payload["manifest_path"], HSIC_GATE_KEY: summary, "hsic_residual_scan": payload}


def inspect_hsic_residual_scan_manifest(manifest: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(manifest))
    if not payload:
        raise FileNotFoundError(f"missing or invalid HSIC residual scan manifest: {manifest}")
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return hsic_residual_scan_summary(payload, manifest_path=manifest)
