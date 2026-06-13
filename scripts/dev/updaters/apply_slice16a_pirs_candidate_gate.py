from __future__ import annotations

import py_compile
import sys
import zipfile
from pathlib import Path
from textwrap import dedent

SLICE = "16A"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/pirs/run_candidates.py",
    "src/pegasus/workflows/pirs_candidates.py",
    "tests/unit/test_slice16a_pirs_candidate_gate.py",
    "tests/integration/test_slice16a_pirs_candidate_gate_from_run_bundle.py",
    "scripts/dev/audits/audit_slice16a_pirs_candidate_gate.py",
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
    raise SystemExit(f"[slice16a] {message}")


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


def preflight() -> None:
    required = (
        "src/pegasus/pirs/schemas.py",
        "src/pegasus/pirs/field_selection.py",
        "src/pegasus/output/bundle.py",
        "src/pegasus/efg/promotion_apply.py",
        "src/pegasus/workflows/efg_apply.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    schemas = read("src/pegasus/pirs/schemas.py")
    if "class FieldCandidate" not in schemas or "quarantined_descriptive" not in schemas:
        fail("PIRS FieldCandidate schema lacks expected permission states")
    selection = read("src/pegasus/pirs/field_selection.py")
    if "def select_fields_for_pirs" not in selection:
        fail("PIRS field selection API missing")
    compile_text = read("src/pegasus/workflows/compile.py")
    if compile_text.count("def run_compile(") != 1 or "def _run_compile_impl(" not in compile_text:
        fail("compile.py public/private run_compile boundary is not in the Slice 13C shape")


def write_run_candidates_module() -> None:
    code = r'''
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from pegasus.pirs.schemas import FieldCandidate


class PIRSCandidateGateError(ValueError):
    """Raised when PIRS run-candidate extraction cannot inspect a run bundle."""


@dataclass(frozen=True)
class PIRSRunCandidateGateResult:
    run_dir: str
    candidate_count: int
    rejected_count: int
    candidates: tuple[FieldCandidate, ...]
    rejected: tuple[dict[str, Any], ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "run_dir": self.run_dir,
            "candidate_count": self.candidate_count,
            "rejected_count": self.rejected_count,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "rejected": list(self.rejected),
            "model_boundary": {
                "metadata_only_fields_model_eligible": False,
                "quarantined_descriptive_fields_model_eligible": False,
                "dashboard_unsafe_fields_model_eligible": False,
                "placeholder_q_tensor_fields_model_eligible": False,
            },
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _parse_json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return default
        return parsed
    return default


def _as_list(value: Any) -> list[Any]:
    parsed = _parse_json_cell(value, [])
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    if parsed in (None, ""):
        return []
    return [parsed]


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_json_cell(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy_false(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"false", "0", "no", "n", "blocked"}


def _role_for_field(field: dict[str, Any]) -> Literal["outcome", "covariate", "offset"]:
    roles = {str(value) for value in _as_list(field.get("role"))}
    name = str(field.get("name") or field.get("field_id") or "").lower()
    unit = str(field.get("unit") or "").lower()
    kind = str(field.get("kind") or "")
    if "offset" in roles or "denominator" in roles or "population" in name or unit in {"person", "persons", "population"}:
        return "offset"
    if kind in {"extensive_measure", "intensive_density", "marked_functional"}:
        return "outcome"
    return "covariate"


def _candidate_utility(field: dict[str, Any], q: dict[str, Any]) -> float:
    n_eff = _safe_float(q.get("n_eff"), 0.0) or 0.0
    missingness = _safe_float(q.get("missingness"), 0.0) or 0.0
    provenance_risk = _safe_float(q.get("provenance_risk"), 0.0) or 0.0
    utility = n_eff * max(0.0, 1.0 - min(1.0, missingness)) * max(0.0, 1.0 - min(1.0, provenance_risk))
    if str(field.get("kind")) == "intensive_density":
        utility *= 1.05
    return float(utility)


def pirs_candidate_rejection_reason(field: dict[str, Any], q: dict[str, Any] | None) -> str | None:
    field_id = str(field.get("field_id") or "")
    state = str(field.get("state") or "")
    materialization_state = str(field.get("materialization_state") or "")
    dashboard_safe = field.get("dashboard_safe")
    warnings = {str(value) for value in _as_list(field.get("warnings"))}
    if not field_id:
        return "missing_field_id"
    if materialization_state == "metadata_only":
        return "metadata_only_field_not_model_eligible"
    if state in {"illegal_excluded", "blocked", "quarantined_descriptive"}:
        return f"q_state_{state}_not_model_eligible"
    if _truthy_false(dashboard_safe):
        return "dashboard_unsafe_not_model_eligible"
    if q is None:
        return "missing_q_tensor_row"
    q_state = str(q.get("state") or state)
    if q_state in {"illegal_excluded", "blocked", "quarantined_descriptive"}:
        return f"q_state_{q_state}_not_model_eligible"
    q_warnings = {str(value) for value in _as_list(q.get("warnings"))}
    if "q_tensor_placeholder_no_numerical_tensor" in q_warnings or "efg_promotion_metadata_only" in q_warnings:
        return "placeholder_q_tensor_not_model_eligible"
    n_eff = _safe_float(q.get("n_eff"), 0.0) or 0.0
    if n_eff <= 0.0:
        return "nonpositive_n_eff_not_model_eligible"
    missingness = _safe_float(q.get("missingness"), 0.0)
    if missingness is not None and missingness >= 1.0:
        return "complete_missingness_not_model_eligible"
    if "efg_promotion_metadata_only" in warnings:
        return "metadata_only_warning_not_model_eligible"
    return None


def pirs_candidate_from_rows(field: dict[str, Any], q: dict[str, Any]) -> FieldCandidate:
    return FieldCandidate(
        field_id=str(field["field_id"]),
        role=_role_for_field(field),
        utility=_candidate_utility(field, q),
        q_state=str(q.get("state") or field.get("state") or "verified"),  # type: ignore[arg-type]
        carrier=str(field.get("carrier") or "unknown"),
        unit=str(field.get("unit") or "unknown"),
        support=_as_dict(field.get("support_json") or field.get("support")),
        variance=_safe_float(q.get("cv"), None),
        warnings=tuple(str(value) for value in sorted(set(_as_list(field.get("warnings")) + _as_list(q.get("warnings"))))),
        provenance=tuple(str(value) for value in _as_list(field.get("provenance"))),
    )


def build_pirs_candidates_from_run(run_dir: str | Path) -> PIRSRunCandidateGateResult:
    root = Path(run_dir)
    v_rows = _read_rows(root / "V_fields.parquet")
    q_rows = _read_rows(root / "Q_tensor.parquet")
    q_by_id = {str(row.get("field_id")): row for row in q_rows if row.get("field_id") is not None}
    candidates: list[FieldCandidate] = []
    rejected: list[dict[str, Any]] = []
    for field in v_rows:
        field_id = str(field.get("field_id") or "")
        q = q_by_id.get(field_id)
        reason = pirs_candidate_rejection_reason(field, q)
        if reason is not None:
            rejected.append({
                "field_id": field_id,
                "reason": reason,
                "state": field.get("state"),
                "materialization_state": field.get("materialization_state"),
                "dashboard_safe": field.get("dashboard_safe"),
            })
            continue
        assert q is not None
        candidates.append(pirs_candidate_from_rows(field, q))
    return PIRSRunCandidateGateResult(
        run_dir=str(root),
        candidate_count=len(candidates),
        rejected_count=len(rejected),
        candidates=tuple(candidates),
        rejected=tuple(rejected),
    )


def write_pirs_candidate_manifest(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    result = build_pirs_candidates_from_run(root)
    output_path = Path(output) if output is not None else root / "Tables" / "pirs_field_candidates.json"
    manifest = result.as_manifest()
    manifest["manifest_path"] = str(output_path)
    _write_json(output_path, manifest)
    return manifest


def attach_pirs_candidate_gate_to_run(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = write_pirs_candidate_manifest(run_dir=root, output=output)
    summary = {
        "schema_version": "1.0",
        "gate": "pirs_candidate_gate",
        "status": "evaluated",
        "candidate_count": manifest["candidate_count"],
        "rejected_count": manifest["rejected_count"],
        "manifest_path": manifest["manifest_path"],
        "metadata_only_fields_model_eligible": False,
        "quarantined_descriptive_fields_model_eligible": False,
    }
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / rel
        if not path.exists():
            continue
        payload = _load_json(path)
        payload["pirs_candidate_gate"] = summary
        _write_json(path, payload)
    return summary
'''
    write("src/pegasus/pirs/run_candidates.py", code)


def write_workflow_module() -> None:
    code = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.run_candidates import attach_pirs_candidate_gate_to_run, write_pirs_candidate_manifest


def run_build_pirs_candidates_from_run(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return write_pirs_candidate_manifest(run_dir=run_dir, output=output)


def run_attach_pirs_candidate_gate(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return attach_pirs_candidate_gate_to_run(run_dir=run_dir, output=output)
'''
    write("src/pegasus/workflows/pirs_candidates.py", code)


def write_tests() -> None:
    unit = r'''
from __future__ import annotations

from pegasus.pirs.run_candidates import pirs_candidate_from_rows, pirs_candidate_rejection_reason


def _field(**updates):
    payload = {
        "field_id": "field_verified",
        "name": "Verified outcome",
        "kind": "extensive_measure",
        "carrier": "Deaths",
        "unit": "counts",
        "state": "verified",
        "dashboard_safe": "True",
        "materialization_state": "materialized",
        "support_json": '{"years":[2020]}',
        "warnings": "[]",
        "provenance": '["fixture"]',
    }
    payload.update(updates)
    return payload


def _q(**updates):
    payload = {
        "field_id": "field_verified",
        "n_eff": 50.0,
        "missingness": 0.0,
        "provenance_risk": 0.1,
        "cv": 0.2,
        "state": "verified",
        "warnings": "[]",
    }
    payload.update(updates)
    return payload


def test_slice16a_rejects_metadata_only_and_quarantined_promotions() -> None:
    reason = pirs_candidate_rejection_reason(
        _field(materialization_state="metadata_only", state="quarantined_descriptive", dashboard_safe="False"),
        _q(state="quarantined_descriptive", warnings='["q_tensor_placeholder_no_numerical_tensor"]'),
    )
    assert reason == "metadata_only_field_not_model_eligible"


def test_slice16a_rejects_placeholder_q_tensor_even_when_field_state_is_relaxed() -> None:
    reason = pirs_candidate_rejection_reason(
        _field(),
        _q(warnings='["q_tensor_placeholder_no_numerical_tensor"]'),
    )
    assert reason == "placeholder_q_tensor_not_model_eligible"


def test_slice16a_builds_field_candidate_for_verified_numeric_field() -> None:
    field = _field()
    q = _q()
    assert pirs_candidate_rejection_reason(field, q) is None
    candidate = pirs_candidate_from_rows(field, q)
    assert candidate.field_id == "field_verified"
    assert candidate.role == "outcome"
    assert candidate.q_state == "verified"
    assert candidate.utility > 0.0
'''
    integration = r'''
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.pirs.field_selection import select_fields_for_pirs
from pegasus.pirs.run_candidates import attach_pirs_candidate_gate_to_run, build_pirs_candidates_from_run


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _append_replace(path: Path, rows: list[dict], id_column: str) -> None:
    schema = pq.read_schema(path)
    existing = pq.read_table(path).to_pylist()
    ids = {str(row[id_column]) for row in rows}
    kept = [row for row in existing if str(row.get(id_column)) not in ids]
    shaped = [{name: row.get(name) for name in schema.names} for row in kept + rows]
    table = pa.Table.from_pylist(shaped, schema=schema)
    pq.write_table(table, path)


def _v(field_id: str, *, state: str, dashboard_safe: str, materialization_state: str, unit: str = "counts") -> dict:
    return {
        "field_id": field_id,
        "name": field_id,
        "kind": "extensive_measure",
        "carrier": "Deaths",
        "unit": unit,
        "aggregation": "additive" if unit == "counts" else "non_aggregable",
        "role": _json(["source_field"]),
        "source": _json(["SIM-DO"]),
        "support_json": _json({"years": [2020], "column": field_id}),
        "axes_json": _json({"period": "year"}),
        "operator": None,
        "provenance": _json(["fixture"]),
        "state": state,
        "dashboard_safe": dashboard_safe,
        "warnings": _json(["efg_promotion_metadata_only"] if materialization_state == "metadata_only" else []),
        "lineage_hash": field_id,
        "registry_hash": "registry",
        "materialization_state": materialization_state,
        "path": None,
    }


def _q(field_id: str, *, state: str, warnings: list[str], n_eff: float = 50.0) -> dict:
    return {
        "field_id": field_id,
        "n_events": 50.0,
        "n_denom": 100.0,
        "n_eff": n_eff,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": 0.0,
        "cv": 0.2,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.1,
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": state,
        "dashboard_safe": "False" if state == "quarantined_descriptive" else "True",
        "warnings": _json(warnings),
        "computed_at": "2026-06-13T00:00:00+00:00",
        "q_schema_version": "1.0",
    }


def test_slice16a_candidate_gate_rejects_metadata_only_promoted_fields_from_run_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    verified = "sim_verified_count"
    promoted = "efg_substrate__SIM_DO__underlying_icd_norm"
    _append_replace(run_dir / "V_fields.parquet", [
        _v(verified, state="verified", dashboard_safe="True", materialization_state="materialized"),
        _v(promoted, state="quarantined_descriptive", dashboard_safe="False", materialization_state="metadata_only", unit="ICD10"),
    ], "field_id")
    _append_replace(run_dir / "Q_tensor.parquet", [
        _q(verified, state="verified", warnings=[]),
        _q(promoted, state="quarantined_descriptive", warnings=["efg_promotion_metadata_only", "q_tensor_placeholder_no_numerical_tensor"], n_eff=0.0),
    ], "field_id")
    # Keep the output bundle validator satisfied for the added field rows.
    _append_replace(run_dir / "VariableDictionary.parquet", [
        {"field_id": verified, "display_name": verified, "technical_name": verified, "definition": "verified", "estimand_label": "count", "source_systems": _json(["SIM-DO"]), "carrier": "Deaths", "unit": "counts", "support_description": _json({}), "axis_description": _json({}), "provenance_description": _json(["fixture"]), "state": "verified", "dashboard_safe": "True", "interpretation_warning": "fixture"},
        {"field_id": promoted, "display_name": promoted, "technical_name": promoted, "definition": "metadata-only", "estimand_label": "metadata_only", "source_systems": _json(["SIM-DO"]), "carrier": "Deaths", "unit": "ICD10", "support_description": _json({}), "axis_description": _json({}), "provenance_description": _json(["fixture"]), "state": "quarantined_descriptive", "dashboard_safe": "False", "interpretation_warning": "not model eligible"},
    ], "field_id")

    assert validate_output_bundle(run_dir=str(run_dir)).ok
    result = build_pirs_candidates_from_run(run_dir)
    assert [candidate.field_id for candidate in result.candidates] == [verified]
    rejected = {row["field_id"]: row["reason"] for row in result.rejected}
    assert rejected[promoted] == "metadata_only_field_not_model_eligible"
    selection = select_fields_for_pirs(list(result.candidates), budget="fast")
    selected = {selection.selected_outcome.field_id if selection.selected_outcome else None, *[c.field_id for c in selection.selected_covariates]}
    assert promoted not in selected

    summary = attach_pirs_candidate_gate_to_run(run_dir=run_dir)
    assert summary["candidate_count"] == 1
    assert (run_dir / "Tables" / "pirs_field_candidates.json").exists()
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["pirs_candidate_gate"]["candidate_count"] == 1
'''
    write("tests/unit/test_slice16a_pirs_candidate_gate.py", unit)
    write("tests/integration/test_slice16a_pirs_candidate_gate_from_run_bundle.py", integration)


def write_audit() -> None:
    code = r'''
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.pirs.field_selection import select_fields_for_pirs
from pegasus.pirs.run_candidates import attach_pirs_candidate_gate_to_run, build_pirs_candidates_from_run


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _append_replace(path: Path, rows: list[dict], id_column: str) -> None:
    schema = pq.read_schema(path)
    existing = pq.read_table(path).to_pylist()
    ids = {str(row[id_column]) for row in rows}
    kept = [row for row in existing if str(row.get(id_column)) not in ids]
    shaped = [{name: row.get(name) for name in schema.names} for row in kept + rows]
    pq.write_table(pa.Table.from_pylist(shaped, schema=schema), path)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must still contain exactly one _run_compile_impl")
    if _count_defs(Path("src/pegasus/pirs/run_candidates.py"), "build_pirs_candidates_from_run") != 1:
        errors.append("PIRS run candidate extraction API missing")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        verified = "audit_verified_count"
        metadata = "efg_substrate__audit_metadata_only"
        _append_replace(run_dir / "V_fields.parquet", [
            {"field_id": verified, "name": verified, "kind": "extensive_measure", "carrier": "Deaths", "unit": "counts", "aggregation": "additive", "role": _json(["source_field"]), "source": _json(["SIM-DO"]), "support_json": _json({}), "axes_json": _json({}), "operator": None, "provenance": _json(["fixture"]), "state": "verified", "dashboard_safe": "True", "warnings": _json([]), "lineage_hash": verified, "registry_hash": "registry", "materialization_state": "materialized", "path": None},
            {"field_id": metadata, "name": metadata, "kind": "observer_proxy", "carrier": "Deaths", "unit": "ICD10", "aggregation": "non_aggregable", "role": _json(["diagnostic_topology"]), "source": _json(["SIM-DO"]), "support_json": _json({}), "axes_json": _json({}), "operator": None, "provenance": _json(["fixture"]), "state": "quarantined_descriptive", "dashboard_safe": "False", "warnings": _json(["efg_promotion_metadata_only"]), "lineage_hash": metadata, "registry_hash": "registry", "materialization_state": "metadata_only", "path": None},
        ], "field_id")
        _append_replace(run_dir / "Q_tensor.parquet", [
            {"field_id": verified, "n_events": 10.0, "n_denom": 20.0, "n_eff": 10.0, "cov_S": 1.0, "cov_T": 1.0, "missingness": 0.0, "zero_inflation": 0.0, "denom_fragility": 0.0, "cv": 0.2, "moran_i": None, "temporal_roughness": None, "spatial_entropy": None, "provenance_risk": 0.1, "race_axis_source": None, "race_axis_target": None, "missing_race_share": None, "emission_prior_strength": None, "race_bridge_cv": None, "sensitivity_width": None, "bridge_mode": None, "state": "verified", "dashboard_safe": "True", "warnings": _json([]), "computed_at": "2026-06-13T00:00:00+00:00", "q_schema_version": "1.0"},
            {"field_id": metadata, "n_events": 0.0, "n_denom": 0.0, "n_eff": 0.0, "cov_S": 0.0, "cov_T": 0.0, "missingness": 1.0, "zero_inflation": 0.0, "denom_fragility": 1.0, "cv": None, "moran_i": None, "temporal_roughness": None, "spatial_entropy": None, "provenance_risk": 0.75, "race_axis_source": None, "race_axis_target": None, "missing_race_share": None, "emission_prior_strength": None, "race_bridge_cv": None, "sensitivity_width": None, "bridge_mode": None, "state": "quarantined_descriptive", "dashboard_safe": "False", "warnings": _json(["q_tensor_placeholder_no_numerical_tensor"]), "computed_at": "2026-06-13T00:00:00+00:00", "q_schema_version": "1.0"},
        ], "field_id")
        result = build_pirs_candidates_from_run(run_dir)
        if [candidate.field_id for candidate in result.candidates] != [verified]:
            errors.append("PIRS candidate gate did not preserve exactly the verified candidate")
        rejected = {row["field_id"]: row["reason"] for row in result.rejected}
        if rejected.get(metadata) != "metadata_only_field_not_model_eligible":
            errors.append("metadata-only promoted field was not rejected before PIRS selection")
        selection = select_fields_for_pirs(list(result.candidates), budget="fast")
        selected = {selection.selected_outcome.field_id if selection.selected_outcome else None, *[c.field_id for c in selection.selected_covariates]}
        if metadata in selected:
            errors.append("metadata-only promoted field entered PIRS selection")
        summary = attach_pirs_candidate_gate_to_run(run_dir=run_dir)
        if summary.get("candidate_count") != 1 or summary.get("rejected_count", 0) < 1:
            errors.append("PIRS candidate gate summary has invalid counts")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16A PIRS candidate gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write("scripts/dev/audits/audit_slice16a_pirs_candidate_gate.py", code)


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    sys.path.insert(0, str(ROOT / "src"))
    from pegasus.pirs.run_candidates import pirs_candidate_rejection_reason
    reason = pirs_candidate_rejection_reason(
        {"field_id": "x", "state": "quarantined_descriptive", "materialization_state": "metadata_only", "dashboard_safe": "False"},
        {"field_id": "x", "state": "quarantined_descriptive", "warnings": "[]", "n_eff": 0},
    )
    if reason != "metadata_only_field_not_model_eligible":
        fail("self-validate metadata-only rejection failed")


def main() -> None:
    preflight()
    write_run_candidates_module()
    write_workflow_module()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 16A updater applied: PIRS candidate gate from run bundle.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
