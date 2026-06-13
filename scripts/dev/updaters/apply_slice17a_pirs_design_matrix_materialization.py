from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "17A"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/pirs/design_matrix.py",
    "src/pegasus/workflows/pirs_matrix.py",
    "tests/unit/test_slice17a_pirs_design_matrix_materialization.py",
    "tests/integration/test_slice17a_pirs_design_matrix_materialization_integration.py",
    "scripts/dev/audits/audit_slice17a_pirs_design_matrix_materialization.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/",
    "src/pegasus/output/",
    "src/pegasus/efg/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice17a] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if path.startswith(FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_function_count(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/pirs/design_plan.py",
        "src/pegasus/pirs/design_readiness.py",
        "src/pegasus/workflows/pirs_pipeline.py",
        "src/pegasus/workflows/pirs_design.py",
        "src/pegasus/workflows/pirs_readiness.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_function_count(compile_text, "run_compile") != 1:
        fail("compile.py must still contain exactly one public run_compile")
    if top_level_function_count(compile_text, "_run_compile_impl") != 1:
        fail("compile.py must still contain exactly one private _run_compile_impl")
    if "pirs_design_readiness_gate" not in read("src/pegasus/pirs/design_readiness.py"):
        fail("Slice 16D design-readiness gate not detected")
    if "run_pirs_planning_pipeline" not in read("src/pegasus/workflows/pirs_pipeline.py"):
        fail("Slice 16F planning pipeline not detected")


def design_matrix_module() -> str:
    return dedent(r'''
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


    def _q_value_vector(row: Mapping[str, Any]) -> list[float] | None:
        for key in VALUE_VECTOR_KEYS:
            vector = _numeric_vector(row.get(key))
            if vector is not None:
                return vector
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
        matrix_rows: list[dict[str, Any]] = []
        for i in range(row_count):
            row: dict[str, Any] = {"row_id": i, "intercept": 1.0}
            for spec in specs:
                row[spec.column] = vectors[spec.field_id][i]
            matrix_rows.append(row)
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
    ''')


def workflow_module() -> str:
    return dedent(r'''
    """Workflow wrappers for Slice 17A PIRS design-matrix artifacts."""

    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    from pegasus.pirs.design_matrix import (
        attach_pirs_design_matrix_to_run,
        inspect_pirs_design_matrix_manifest,
        write_pirs_design_matrix_artifact,
    )


    def run_write_pirs_design_matrix(
        *,
        run_dir: str | Path,
        design_plan: str | Path | dict[str, Any] | None = None,
        readiness_manifest: str | Path | dict[str, Any] | None = None,
        output_manifest: str | Path | None = None,
        output_matrix: str | Path | None = None,
        write_matrix: bool = True,
    ) -> dict[str, Any]:
        return write_pirs_design_matrix_artifact(
            run_dir=run_dir,
            design_plan=design_plan,
            readiness_manifest=readiness_manifest,
            output_manifest=output_manifest,
            output_matrix=output_matrix,
            write_matrix=write_matrix,
        )


    def run_attach_pirs_design_matrix_to_run(
        *,
        run_dir: str | Path,
        design_plan: str | Path | dict[str, Any] | None = None,
        readiness_manifest: str | Path | dict[str, Any] | None = None,
        output_manifest: str | Path | None = None,
        output_matrix: str | Path | None = None,
        write_matrix: bool = True,
    ) -> dict[str, Any]:
        return attach_pirs_design_matrix_to_run(
            run_dir=run_dir,
            design_plan=design_plan,
            readiness_manifest=readiness_manifest,
            output_manifest=output_manifest,
            output_matrix=output_matrix,
            write_matrix=write_matrix,
        )


    __all__ = [
        "inspect_pirs_design_matrix_manifest",
        "run_attach_pirs_design_matrix_to_run",
        "run_write_pirs_design_matrix",
    ]
    ''')


def unit_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pegasus.pirs.design_matrix import (
        build_pirs_design_matrix_manifest,
        inspect_pirs_design_matrix_manifest,
        pirs_design_matrix_summary,
    )


    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


    def _write_q(run_dir: Path, rows: list[dict]) -> None:
        pq.write_table(pa.Table.from_pylist(rows), run_dir / "Q_tensor.parquet")


    def test_slice17a_blocks_when_design_readiness_is_not_ready(tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "Tables").mkdir(parents=True)
        _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "planned", "outcome_field_id": "field:y", "covariate_field_ids": ["field:x"]})
        _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "blocked", "ready": False})
        _write_q(run_dir, [{"field_id": "field:y", "values_json": "[1, 2]"}, {"field_id": "field:x", "values_json": "[3, 4]"}])

        payload = build_pirs_design_matrix_manifest(run_dir=run_dir)

        assert payload["status"] == "blocked"
        assert payload["matrix_written"] is False
        assert any(str(reason).startswith("design_readiness_not_ready") for reason in payload["blocking_reasons"])


    def test_slice17a_materializes_numeric_design_matrix_when_ready(tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "Tables").mkdir(parents=True)
        _write_json(
            run_dir / "Tables" / "pirs_design_plan.json",
            {
                "status": "planned",
                "family": "poisson_rate",
                "residual_mode": "cross_fitted",
                "outcome_field_id": "field:y",
                "covariate_field_ids": ["field:x1", "field:x2"],
                "offset_field_id": "field:pop",
            },
        )
        _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "ready", "ready": True})
        _write_q(
            run_dir,
            [
                {"field_id": "field:y", "values_json": "[10, 20, 30]"},
                {"field_id": "field:x1", "values_json": "[1, 2, 3]"},
                {"field_id": "field:x2", "values_json": "[4, 5, 6]"},
                {"field_id": "field:pop", "values_json": "[100, 200, 300]"},
            ],
        )

        payload = build_pirs_design_matrix_manifest(run_dir=run_dir)

        assert payload["status"] == "ready"
        assert payload["design_matrix_state"] == "materialized"
        assert payload["row_count"] == 3
        assert payload["columns"] == ["row_id", "intercept", "response", "covariate_001", "covariate_002", "offset"]
        rows = pq.read_table(run_dir / "Tables" / "pirs_design_matrix.parquet").to_pylist()
        assert rows[0]["response"] == 10.0
        assert rows[0]["intercept"] == 1.0
        summary = pirs_design_matrix_summary(payload)
        assert summary["model_fit_state"] == "not_started"
        inspected = inspect_pirs_design_matrix_manifest(run_dir / "Tables" / "pirs_design_matrix_manifest.json")
        assert inspected["status"] == "ready"
    ''')


def integration_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run


    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


    def test_slice17a_attach_blocks_empty_bundle_without_numerical_values(tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        create_empty_output_bundle(run_dir)
        _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "blocked", "warnings": ["empty_bundle"]})
        _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "blocked", "ready": False})

        result = run_attach_pirs_design_matrix_to_run(run_dir=run_dir)

        gate = result["pirs_design_matrix_gate"]
        assert gate["status"] == "blocked"
        assert gate["matrix_written"] is False
        assert (run_dir / "Tables" / "pirs_design_matrix_manifest.json").exists()
        assert not (run_dir / "Tables" / "pirs_design_matrix.parquet").exists()
        p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
        assert p_vector["pirs_design_matrix_gate"]["status"] == "blocked"


    def test_slice17a_attach_writes_matrix_for_ready_synthetic_bundle(tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        create_empty_output_bundle(run_dir)
        _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "planned", "outcome_field_id": "y", "covariate_field_ids": ["x"]})
        _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "ready", "ready": True})
        pq.write_table(pa.Table.from_pylist([
            {"field_id": "y", "values_json": "[1, 0, 1]"},
            {"field_id": "x", "values_json": "[10, 20, 30]"},
        ]), run_dir / "Q_tensor.parquet")

        result = run_attach_pirs_design_matrix_to_run(run_dir=run_dir)

        gate = result["pirs_design_matrix_gate"]
        assert gate["status"] == "ready"
        assert gate["row_count"] == 3
        assert (run_dir / "Tables" / "pirs_design_matrix.parquet").exists()
        rows = pq.read_table(run_dir / "Tables" / "pirs_design_matrix.parquet").to_pylist()
        assert rows[2]["covariate_001"] == 30.0
    ''')


def audit_script() -> str:
    return dedent(r'''
    from __future__ import annotations

    import ast
    import json
    import tempfile
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run


    def _count_defs(path: Path, name: str) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


    def main() -> int:
        errors: list[str] = []
        if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
            errors.append("compile.py must still contain exactly one public run_compile")
        if _count_defs(Path("src/pegasus/pirs/design_matrix.py"), "build_pirs_design_matrix_manifest") != 1:
            errors.append("design matrix builder API missing")
        if _count_defs(Path("src/pegasus/workflows/pirs_matrix.py"), "run_attach_pirs_design_matrix_to_run") != 1:
            errors.append("design matrix workflow API missing")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            create_empty_output_bundle(run_dir)
            _write_json(run_dir / "Tables" / "pirs_design_plan.json", {"status": "planned", "outcome_field_id": "y", "covariate_field_ids": ["x"]})
            _write_json(run_dir / "Tables" / "pirs_design_readiness.json", {"status": "ready", "ready": True})
            pq.write_table(pa.Table.from_pylist([
                {"field_id": "y", "values_json": "[1, 2]"},
                {"field_id": "x", "values_json": "[3, 4]"},
            ]), run_dir / "Q_tensor.parquet")
            result = run_attach_pirs_design_matrix_to_run(run_dir=run_dir)
            gate = result.get("pirs_design_matrix_gate", {})
            if gate.get("status") != "ready":
                errors.append("ready synthetic bundle did not produce ready design matrix gate")
            if not (run_dir / "Tables" / "pirs_design_matrix.parquet").exists():
                errors.append("ready synthetic bundle did not write pirs_design_matrix.parquet")
            p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
            if "pirs_design_matrix_gate" not in p_vector:
                errors.append("P_vector.json missing pirs_design_matrix_gate")

        payload = {"ok": not errors, "errors": errors}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        if errors:
            return 1
        print("AUDIT PASSED: Slice 17A PIRS design matrix materialization boundary")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    ''')


def write_files() -> None:
    write("src/pegasus/pirs/design_matrix.py", design_matrix_module())
    write("src/pegasus/workflows/pirs_matrix.py", workflow_module())
    write("tests/unit/test_slice17a_pirs_design_matrix_materialization.py", unit_test())
    write("tests/integration/test_slice17a_pirs_design_matrix_materialization_integration.py", integration_test())
    write("scripts/dev/audits/audit_slice17a_pirs_design_matrix_materialization.py", audit_script())


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pegasus.pirs.design_matrix import build_pirs_design_matrix_manifest
    from pegasus.workflows.pirs_matrix import run_attach_pirs_design_matrix_to_run
    if not callable(build_pirs_design_matrix_manifest) or not callable(run_attach_pirs_design_matrix_to_run):
        fail("post-validate: design matrix APIs not callable")


def main() -> None:
    preflight()
    write_files()
    self_validate()
    print("Slice 17A updater applied: PIRS design matrix materialization boundary added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
