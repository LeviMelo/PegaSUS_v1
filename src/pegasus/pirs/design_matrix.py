
"""PIRS design-matrix materialization boundary.

Slice 17A is the first numerical artifact boundary after the PIRS planning
stack. It may materialize a numeric design matrix only when the design
readiness artifact explicitly says the plan is ready and every selected
field has tensor-backed vector values in Q_tensor. It does not fit models,
create residuals, run HSIC, or mutate compile outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DESIGN_PLAN = Path("Tables") / "pirs_design_plan.json"
DEFAULT_READINESS = Path("Tables") / "pirs_design_readiness.json"
DEFAULT_MATRIX_MANIFEST = Path("Tables") / "pirs_design_matrix_manifest.json"
DEFAULT_MATRIX = Path("Tables") / "pirs_design_matrix.parquet"
DEFAULT_SUPPORT_INDEX = Path("Tables") / "support_index.parquet"
MATRIX_GATE_KEY = "pirs_design_matrix_gate"
JSON_ATTACH_TARGETS: tuple[str, ...] = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)
VALUE_VECTOR_KEYS: tuple[str, ...] = (
    "values_json",
    "value_vector_json",
    "tensor_values_json",
    "observations_json",
    "numeric_values_json",
)


class PIRSDesignMatrixError(RuntimeError):
    """Raised when the design matrix boundary receives malformed input."""


@dataclass(frozen=True)
class MatrixFieldSpec:
    field_id: str
    role: str
    column: str
    required: bool = True

    def as_manifest(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "role": self.role,
            "column": self.column,
            "required": self.required,
        }


def _safe_column_token(value: Any) -> str:
    text = str(value).strip()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    return out.strip("_") or "unknown"


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
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)
    return path


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [value]
        if isinstance(decoded, list):
            return decoded
        return [decoded]
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if item not in (None, "")]


def _json_vector(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, list):
            return decoded
    return None


def _numeric_vector(value: Any) -> list[float] | None:
    raw = _json_vector(value)
    if raw is None:
        return None
    out: list[float] = []
    for item in raw:
        if item in (None, ""):
            return None
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _q_value_vector(row: Mapping[str, Any]) -> list[float | None] | None:
    """Parse a Q_tensor value vector, tolerating null cells.

    Panel fields legitimately have null cells (a cause absent in a given
    municipality-year). Nulls are preserved (not coerced to 0) so missingness is
    handled honestly downstream; only a genuinely non-numeric non-null entry
    disqualifies the vector.
    """
    for key in VALUE_VECTOR_KEYS:
        raw = _json_vector(row.get(key))
        if raw is None:
            continue
        out: list[float | None] = []
        ok = True
        for item in raw:
            if item is None or item == "":
                out.append(None)
                continue
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            return out
    return None


def _readiness_ready(readiness: Mapping[str, Any]) -> bool:
    if readiness.get("status") == "ready" or readiness.get("ready") is True:
        return True
    for key in ("pirs_design_readiness_gate", "summary", "gate"):
        value = readiness.get(key)
        if isinstance(value, Mapping) and (value.get("status") == "ready" or value.get("ready") is True):
            return True
    return False


def _readiness_status(readiness: Mapping[str, Any]) -> str:
    if readiness.get("status") is not None:
        return str(readiness.get("status"))
    for key in ("pirs_design_readiness_gate", "summary", "gate"):
        value = readiness.get(key)
        if isinstance(value, Mapping) and value.get("status") is not None:
            return str(value.get("status"))
    return "missing"


def _selected_field_specs(plan: Mapping[str, Any]) -> list[MatrixFieldSpec]:
    specs: list[MatrixFieldSpec] = []
    outcome = plan.get("outcome_field_id") or plan.get("selected_outcome_field_id")
    if outcome not in (None, ""):
        specs.append(MatrixFieldSpec(field_id=str(outcome), role="outcome", column="response", required=True))
    for i, field_id in enumerate(_string_list(plan.get("covariate_field_ids") or plan.get("selected_covariate_field_ids")), start=1):
        specs.append(MatrixFieldSpec(field_id=field_id, role="covariate", column=f"covariate_{i:03d}", required=True))
    offset = plan.get("offset_field_id") or plan.get("selected_offset_field_id")
    if offset not in (None, ""):
        specs.append(MatrixFieldSpec(field_id=str(offset), role="offset", column="offset", required=False))
    return specs


def _spatial_effect_mode(plan: Mapping[str, Any]) -> str:
    value = plan.get("spatial_effect_mode")
    if value in (None, ""):
        spatial = plan.get("spatial_effect")
        if isinstance(spatial, Mapping):
            value = spatial.get("mode")
    return str(value or "none")


def _support_rows(root: Path) -> list[dict[str, Any]]:
    return _read_rows(root / DEFAULT_SUPPORT_INDEX)


def _support_category(row: Mapping[str, Any], *, mode: str) -> str | None:
    if mode == "UF_FE":
        for key in ("uf", "state", "ibge_uf_cod2", "datasus_uf_prefix"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        for key in ("municipality", "municipality_cod6", "mun_residence_cod6", "mun_occurrence_cod6", "ibge_cod7"):
            value = row.get(key)
            if value not in (None, ""):
                text = str(value)
                return text[:2] if len(text) >= 2 else text
    if mode == "municipality_FE":
        for key in ("municipality", "municipality_cod6", "mun_residence_cod6", "mun_occurrence_cod6", "ibge_cod7"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _add_fixed_effect_columns(
    *,
    matrix_rows: list[dict[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> tuple[list[MatrixFieldSpec], list[str], dict[str, Any]]:
    if mode not in {"UF_FE", "municipality_FE"}:
        return [], [], {"mode": mode, "executed": mode == "none", "columns": []}
    if not support_rows:
        return [], [f"{mode.lower()}_requires_support_index"], {"mode": mode, "executed": False, "columns": []}

    categories_by_row: list[str | None] = []
    for row in matrix_rows:
        try:
            support = support_rows[int(row.get("row_id", -1))]
        except (TypeError, ValueError, IndexError):
            support = {}
        categories_by_row.append(_support_category(support, mode=mode))
    missing = sum(1 for value in categories_by_row if value in (None, ""))
    if missing:
        return [], [f"{mode.lower()}_support_category_missing:{missing}_rows"], {"mode": mode, "executed": False, "columns": []}
    categories = sorted({str(value) for value in categories_by_row if value not in (None, "")})
    if len(categories) <= 1:
        return [], [f"{mode.lower()}_single_category_no_dummy_columns"], {
            "mode": mode,
            "executed": True,
            "columns": [],
            "baseline_category": categories[0] if categories else None,
        }
    baseline = categories[0]
    columns: list[str] = []
    specs: list[MatrixFieldSpec] = []
    for category in categories[1:]:
        column = f"spatial_{mode.lower()}_{_safe_column_token(category)}"
        columns.append(column)
        specs.append(MatrixFieldSpec(field_id=f"{mode}:{category}", role="spatial_effect", column=column, required=True))
        for row, row_category in zip(matrix_rows, categories_by_row, strict=True):
            row[column] = 1.0 if row_category == category else 0.0
    return specs, [], {
        "mode": mode,
        "executed": True,
        "columns": columns,
        "baseline_category": baseline,
        "category_count": len(categories),
    }


def _blocking_manifest(
    *,
    run_dir: Path,
    output_manifest: Path,
    output_matrix: Path,
    design_plan_path: Path,
    readiness_path: Path,
    reasons: Sequence[str],
    plan: Mapping[str, Any] | None = None,
    readiness: Mapping[str, Any] | None = None,
    field_specs: Sequence[MatrixFieldSpec] = (),
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slice": "17A",
        "artifact": "pirs_design_matrix_manifest",
        "status": "blocked",
        "design_matrix_state": "blocked_until_tensor_backed_values",
        "run_dir": str(run_dir),
        "manifest_path": str(output_manifest),
        "matrix_path": str(output_matrix),
        "matrix_written": False,
        "row_count": 0,
        "column_count": 0,
        "columns": [],
        "field_specs": [spec.as_manifest() for spec in field_specs],
        "blocking_reasons": list(dict.fromkeys(str(reason) for reason in reasons)),
        "source_design_plan": str(design_plan_path),
        "source_design_readiness": str(readiness_path),
        "readiness_status": _readiness_status(readiness or {}),
        "design_plan_status": (plan or {}).get("status"),
        "model_fit_state": "not_started",
        "residual_state": "not_started",
        "hsic_state": "not_started",
        "non_model_fit_artifact": True,
    }


def build_pirs_design_matrix_manifest(
    *,
    run_dir: str | Path,
    design_plan: str | Path | Mapping[str, Any] | None = None,
    readiness_manifest: str | Path | Mapping[str, Any] | None = None,
    output_manifest: str | Path | None = None,
    output_matrix: str | Path | None = None,
    write_matrix: bool = True,
) -> dict[str, Any]:
    """Build a design matrix manifest and optionally write a numeric matrix."""
    root = Path(run_dir)
    design_plan_path = Path(design_plan) if isinstance(design_plan, (str, Path)) else root / DEFAULT_DESIGN_PLAN
    readiness_path = Path(readiness_manifest) if isinstance(readiness_manifest, (str, Path)) else root / DEFAULT_READINESS
    manifest_path = Path(output_manifest) if output_manifest is not None else root / DEFAULT_MATRIX_MANIFEST
    matrix_path = Path(output_matrix) if output_matrix is not None else root / DEFAULT_MATRIX
    plan = dict(design_plan) if isinstance(design_plan, Mapping) else _load_json(design_plan_path)
    readiness = dict(readiness_manifest) if isinstance(readiness_manifest, Mapping) else _load_json(readiness_path)
    specs = _selected_field_specs(plan)
    spatial_mode = _spatial_effect_mode(plan)
    reasons: list[str] = []
    if not specs:
        reasons.append("design_plan_has_no_selected_fields")
    if not _readiness_ready(readiness):
        reasons.append(f"design_readiness_not_ready:{_readiness_status(readiness)}")
    q_rows = _read_rows(root / "Q_tensor.parquet")
    q_by_id = {str(row.get("field_id")): row for row in q_rows if row.get("field_id") not in (None, "")}
    vectors: dict[str, list[float]] = {}
    lengths: set[int] = set()
    for spec in specs:
        q = q_by_id.get(spec.field_id)
        if q is None:
            reasons.append(f"q_tensor_row_missing:{spec.field_id}")
            continue
        vector = _q_value_vector(q)
        if vector is None:
            reasons.append(f"q_tensor_values_missing_or_non_numeric:{spec.field_id}")
            continue
        vectors[spec.field_id] = vector
        lengths.add(len(vector))
    if len(lengths) > 1:
        reasons.append("q_tensor_value_vectors_have_inconsistent_lengths")
    if 0 in lengths:
        reasons.append("q_tensor_value_vectors_empty")
    if reasons:
        payload = _blocking_manifest(
            run_dir=root,
            output_manifest=manifest_path,
            output_matrix=matrix_path,
            design_plan_path=design_plan_path,
            readiness_path=readiness_path,
            reasons=reasons,
            plan=plan,
            readiness=readiness,
            field_specs=specs,
        )
        _write_json(manifest_path, payload)
        return payload

    row_count = next(iter(lengths)) if lengths else 0
    outcome_field_ids = [spec.field_id for spec in specs if spec.role == "outcome"]
    matrix_rows: list[dict[str, Any]] = []
    for i in range(row_count):
        # Complete-case on the outcome: a panel cell with no observed outcome
        # carries no regression information. Covariate nulls are retained and
        # handled by missingness indicators at fit time (MSD §2.3).
        if any(vectors[field_id][i] is None for field_id in outcome_field_ids):
            continue
        row: dict[str, Any] = {"row_id": i, "intercept": 1.0}
        for spec in specs:
            row[spec.column] = vectors[spec.field_id][i]
        matrix_rows.append(row)
    if not matrix_rows:
        payload = _blocking_manifest(
            run_dir=root,
            output_manifest=manifest_path,
            output_matrix=matrix_path,
            design_plan_path=design_plan_path,
            readiness_path=readiness_path,
            reasons=["design_matrix_empty_after_outcome_complete_case"],
            plan=plan,
            readiness=readiness,
            field_specs=specs,
        )
        _write_json(manifest_path, payload)
        return payload
    support_rows = _support_rows(root)
    spatial_specs, spatial_reasons, spatial_manifest = _add_fixed_effect_columns(
        matrix_rows=matrix_rows,
        support_rows=support_rows,
        mode=spatial_mode,
    )
    if spatial_mode == "ICAR":
        spatial_manifest = {
            "mode": "ICAR",
            "executed": False,
            "execution_stage": "model_execution_penalized_irls",
            "adjacency_path": (plan.get("spatial_effect") or {}).get("adjacency_path")
            if isinstance(plan.get("spatial_effect"), Mapping)
            else None,
        }
    if spatial_reasons and spatial_mode in {"UF_FE", "municipality_FE"}:
        payload = _blocking_manifest(
            run_dir=root,
            output_manifest=manifest_path,
            output_matrix=matrix_path,
            design_plan_path=design_plan_path,
            readiness_path=readiness_path,
            reasons=spatial_reasons,
            plan=plan,
            readiness=readiness,
            field_specs=[*specs, *spatial_specs],
        )
        payload["spatial_effect"] = spatial_manifest
        _write_json(manifest_path, payload)
        return payload
    specs = [*specs, *spatial_specs]
    columns = list(matrix_rows[0].keys()) if matrix_rows else ["row_id", "intercept"]
    if write_matrix:
        _write_rows(matrix_path, matrix_rows)
    payload = {
        "schema_version": "1.0",
        "slice": "17A",
        "artifact": "pirs_design_matrix_manifest",
        "status": "ready",
        "design_matrix_state": "materialized" if write_matrix else "ready_not_written",
        "run_dir": str(root),
        "manifest_path": str(manifest_path),
        "matrix_path": str(matrix_path),
        "matrix_written": bool(write_matrix),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "field_specs": [spec.as_manifest() for spec in specs],
        "blocking_reasons": [],
        "source_design_plan": str(design_plan_path),
        "source_design_readiness": str(readiness_path),
        "readiness_status": _readiness_status(readiness),
        "design_plan_status": plan.get("status"),
        "family": plan.get("family"),
        "residual_mode": plan.get("residual_mode"),
        "fold_scheme": plan.get("fold_scheme"),
        "spatial_effect_mode": spatial_mode,
        "spatial_effect": spatial_manifest,
        "model_fit_state": "not_started",
        "residual_state": "not_started",
        "hsic_state": "not_started",
        "non_model_fit_artifact": True,
    }
    _write_json(manifest_path, payload)
    return payload


def pirs_design_matrix_summary(payload: Mapping[str, Any], *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slice": "17A",
        "gate": MATRIX_GATE_KEY,
        "status": payload.get("status"),
        "design_matrix_state": payload.get("design_matrix_state"),
        "matrix_written": bool(payload.get("matrix_written")),
        "row_count": payload.get("row_count", 0),
        "column_count": payload.get("column_count", 0),
        "blocking_reason_count": len(_as_list(payload.get("blocking_reasons"))),
        "matrix_path": payload.get("matrix_path"),
        "manifest_path": str(manifest_path) if manifest_path is not None else payload.get("manifest_path"),
        "model_fit_state": payload.get("model_fit_state", "not_started"),
        "residual_state": payload.get("residual_state", "not_started"),
        "hsic_state": payload.get("hsic_state", "not_started"),
        "non_model_fit_artifact": True,
    }


def attach_pirs_design_matrix_gate_to_run(*, run_dir: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(run_dir)
    for rel in JSON_ATTACH_TARGETS:
        path = root / rel
        payload = _load_json(path)
        payload[MATRIX_GATE_KEY] = dict(summary)
        _write_json(path, payload)


def write_pirs_design_matrix_artifact(
    *,
    run_dir: str | Path,
    design_plan: str | Path | Mapping[str, Any] | None = None,
    readiness_manifest: str | Path | Mapping[str, Any] | None = None,
    output_manifest: str | Path | None = None,
    output_matrix: str | Path | None = None,
    write_matrix: bool = True,
) -> dict[str, Any]:
    payload = build_pirs_design_matrix_manifest(
        run_dir=run_dir,
        design_plan=design_plan,
        readiness_manifest=readiness_manifest,
        output_manifest=output_manifest,
        output_matrix=output_matrix,
        write_matrix=write_matrix,
    )
    summary = pirs_design_matrix_summary(payload, manifest_path=payload.get("manifest_path"))
    payload["summary"] = summary
    _write_json(Path(str(payload["manifest_path"])), payload)
    return payload


def attach_pirs_design_matrix_to_run(
    *,
    run_dir: str | Path,
    design_plan: str | Path | Mapping[str, Any] | None = None,
    readiness_manifest: str | Path | Mapping[str, Any] | None = None,
    output_manifest: str | Path | None = None,
    output_matrix: str | Path | None = None,
    write_matrix: bool = True,
) -> dict[str, Any]:
    payload = write_pirs_design_matrix_artifact(
        run_dir=run_dir,
        design_plan=design_plan,
        readiness_manifest=readiness_manifest,
        output_manifest=output_manifest,
        output_matrix=output_matrix,
        write_matrix=write_matrix,
    )
    summary = dict(payload["summary"])
    attach_pirs_design_matrix_gate_to_run(run_dir=run_dir, summary=summary)
    return {"manifest_path": payload["manifest_path"], MATRIX_GATE_KEY: summary, "design_matrix": payload}


def inspect_pirs_design_matrix_manifest(manifest: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(manifest))
    if not payload:
        raise FileNotFoundError(f"missing or invalid PIRS design matrix manifest: {manifest}")
    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    return pirs_design_matrix_summary(payload, manifest_path=manifest)
