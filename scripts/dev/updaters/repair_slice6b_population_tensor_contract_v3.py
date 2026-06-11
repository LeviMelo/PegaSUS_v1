from __future__ import annotations

"""Repair Slice 6B population tensor metadata/schema contract.

This repair is intentionally full-file for the small population schema and
population attach module, and function-scoped or line-scoped for larger files.
It avoids the v1/v2 brittle assumptions that caused updater failures.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def path(rel: str) -> Path:
    return REPO / rel


def write(rel: str, text: str) -> None:
    p = path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


POPULATION_SCHEMA = r'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pegasus.core.schemas import DenominatorContract


PopulationTensorMode = Literal["independent_denominator", "sim_informed_denominator"]


def official_sidra_anchor_contract(
    *,
    source: str = "SIDRA",
    warnings: list[str] | None = None,
) -> DenominatorContract:
    return DenominatorContract(
        mode="official_sidra_anchor",
        source=source,
        provenance=["official"],
        state="fragile",
        dashboard_safe="warning",
        allowed_for_rates=True,
        warnings=warnings or ["fixture_or_unvalidated_sidra_anchor"],
    )


def blocked_missing_population_contract(
    *,
    reason: str,
) -> DenominatorContract:
    return DenominatorContract(
        mode="blocked_missing",
        source="none",
        provenance=[],
        state="illegal_excluded",
        dashboard_safe=False,
        allowed_for_rates=False,
        warnings=[reason],
    )


@dataclass(frozen=True)
class PopulationTensorRequest:
    mode: PopulationTensorMode
    solver_id: str
    solver_backend: str
    locality_ids: tuple[str, ...]
    periods: tuple[str, ...]
    strata: tuple[str, ...] = ("total",)
    require_sparse: bool = True

    @property
    def n_cells(self) -> int:
        return len(self.locality_ids) * len(self.periods) * len(self.strata)

    def support(self) -> dict[str, Any]:
        return {
            "locality_ids": list(self.locality_ids),
            "periods": list(self.periods),
            "strata": list(self.strata),
            "n_cells": self.n_cells,
        }


@dataclass(frozen=True)
class PopulationTensorDiagnostics:
    n_cells: int
    solver_backend: str
    sparse_jacobian: bool
    reconstruction_uncertainty: float
    denominator_feedback_warning: bool
    dense_abort_threshold: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "solver_backend": self.solver_backend,
            "sparse_jacobian": self.sparse_jacobian,
            "reconstruction_uncertainty": self.reconstruction_uncertainty,
            "denominator_feedback_warning": self.denominator_feedback_warning,
            "dense_abort_threshold": self.dense_abort_threshold,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PopulationTensorResult:
    tensor_id: str
    mode: PopulationTensorMode
    solver_id: str
    solver_backend: str
    sparse_jacobian: bool
    value: float
    unit: str
    locality_id: str
    period: str
    source_anchor_field_id: str
    source_table_id: str
    source_variable_id: str
    source_request_hash: str
    source_metadata_hash: str
    reconstruction_uncertainty: float
    denominator_feedback_warning: bool
    state: str
    warnings: tuple[str, ...]
    diagnostics: PopulationTensorDiagnostics

    def as_manifest(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "mode": self.mode,
            "PopulationTensorMode": self.mode,
            "solver_backend": self.solver_backend,
            "SolverBackend": self.solver_backend,
            "solver_id": self.solver_id,
            "SolverID": self.solver_id,
            "sparse_jacobian": self.sparse_jacobian,
            "SparseJacobian": self.sparse_jacobian,
            "value": self.value,
            "unit": self.unit,
            "locality_id": self.locality_id,
            "period": self.period,
            "source_anchor_field_id": self.source_anchor_field_id,
            "source_table_id": self.source_table_id,
            "source_variable_id": self.source_variable_id,
            "source_request_hash": self.source_request_hash,
            "source_metadata_hash": self.source_metadata_hash,
            "reconstruction_uncertainty": self.reconstruction_uncertainty,
            "denominator_feedback_warning": self.denominator_feedback_warning,
            "DenominatorFeedbackWarning": self.denominator_feedback_warning,
            "state": self.state,
            "warnings": list(self.warnings),
            "diagnostics": self.diagnostics.as_manifest(),
        }
'''


POPULATION_TENSOR_COMPILE_ATTACH = r'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from pegasus.core.hashing import sha256_file
from pegasus.output.population_tensor_bundle import (
    _failed_dense_branch,
    _field,
    _q_row,
    _vd_row,
    _warning_rows,
)
from pegasus.she.population.solvers import solve_population_tensor_from_sidra_anchor

POPULATION_TENSOR_FIELD_PREFIX = "population_tensor_"
POPULATION_TENSOR_WARNING_PREFIX = "population_tensor_"
POPULATION_TENSOR_FAILED_BRANCH_IDS = {"failed_dense_national_population_tensor_above_threshold"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pq.read_table(path).schema
    shaped = [{name: row.get(name) for name in schema.names} for row in rows]
    if shaped:
        table = pa.Table.from_pylist(shaped, schema=schema)
    else:
        table = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    pq.write_table(table, path)


def _append_replace(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    id_column: str,
    remove_ids: set[str] | None = None,
    remove_prefixes: tuple[str, ...] = (),
) -> None:
    remove_ids = remove_ids or set()
    existing = _read_rows(path)

    def keep(row: dict[str, Any]) -> bool:
        value = str(row.get(id_column, ""))
        if value in remove_ids:
            return False
        return not any(value.startswith(prefix) for prefix in remove_prefixes)

    _write_rows_like(path, [row for row in existing if keep(row)] + rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _metadata(*, result, field: dict[str, Any], sidra_facts_path: Path, mode: str) -> dict[str, Any]:
    manifest = result.as_manifest()
    manifest.update(
        {
            "schema_version": "1.0",
            "source_systems": ["SIDRA"],
            "attach_stage": "population_solver",
            "field_id": field["field_id"],
            "field_name": field["name"],
            "requested_population_mode": mode,
            "sidra_facts_path": str(sidra_facts_path),
            "source_hashes": {"sidra_facts": sha256_file(sidra_facts_path)},
            "independent_denominator_mode": result.mode == "independent_denominator",
            "sim_feedback_warning": bool(result.denominator_feedback_warning),
            "dashboard_safe": field.get("dashboard_safe"),
            "materialization_state": field.get("materialization_state"),
            "table_paths": {"diagnostics": "Tables/population_tensor_diagnostics.parquet"},
        }
    )
    return manifest


def attach_population_tensor_compile_fields(
    *,
    run_dir: str | Path,
    sidra_facts_path: str | Path,
    mode: str = "independent_denominator",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    sidra_facts_path = Path(sidra_facts_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not sidra_facts_path.exists():
        raise FileNotFoundError(f"sidra_facts_path does not exist: {sidra_facts_path}")

    result = solve_population_tensor_from_sidra_anchor(sidra_facts_path=sidra_facts_path, mode=mode)
    field = _field(result)
    q_row = _q_row(field, result)
    vd_row = _vd_row(field, result)
    warning_rows = _warning_rows(field, result)
    failed_row = _failed_dense_branch(field)
    meta = _metadata(result=result, field=field, sidra_facts_path=sidra_facts_path, mode=mode)

    _append_replace(run_dir / "V_fields.parquet", [field], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "Q_tensor.parquet", [q_row], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "VariableDictionary.parquet", [vd_row], id_column="field_id", remove_prefixes=(POPULATION_TENSOR_FIELD_PREFIX,))
    _append_replace(run_dir / "Warnings.parquet", warning_rows, id_column="warning_id", remove_prefixes=(POPULATION_TENSOR_WARNING_PREFIX,))
    _append_replace(run_dir / "FailedBranches.parquet", [failed_row], id_column="failed_branch_id", remove_ids=POPULATION_TENSOR_FAILED_BRANCH_IDS)

    (run_dir / "Tables").mkdir(exist_ok=True)
    pl.DataFrame([meta]).write_parquet(run_dir / "Tables" / "population_tensor_diagnostics.parquet")

    for json_name in ["RunConfig.json", "ReproducibilityManifest.json", "P_vector.json"]:
        path = run_dir / json_name
        payload = _load_json(path)
        payload["population_tensor"] = meta
        if json_name == "P_vector.json":
            payload.setdefault("source_systems", [])
            if "SIDRA" not in payload["source_systems"]:
                payload["source_systems"].append("SIDRA")
        _write_json(path, payload)

    return meta
'''


def replace_function(text: str, name: str, replacement: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {name}("):
            start = i
            break
    if start is None:
        raise RuntimeError(f"top-level function not found in population_tensor_bundle.py: {name}")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("class "):
            end = j
            break
    return "".join(lines[:start]) + replacement.rstrip() + "\n\n" + "".join(lines[end:])


FIELD_FUNCTION = r'''
def _field(result: PopulationTensorResult) -> dict[str, Any]:
    support = {
        "PopulationTensorMode": result.mode,
        "SolverBackend": result.solver_backend,
        "SolverID": result.solver_id,
        "SparseJacobian": result.sparse_jacobian,
        "DenominatorFeedbackWarning": result.denominator_feedback_warning,
        "reconstruction_uncertainty": result.reconstruction_uncertainty,
        "locality_id": result.locality_id,
        "period": result.period,
        "n_events": result.value,
        "n_eff": result.value,
        "missingness": 0.0,
        "denom_fragility": result.reconstruction_uncertainty,
        "population_tensor_diagnostics": result.diagnostics.as_manifest(),
    }
    axes = {
        "geography_axis": "IBGE_COD7",
        "time_axis": "year",
        "population_strata_axis": "total",
        "population_tensor_mode": result.mode,
    }
    role = ["population_denominator_tensor", result.mode]
    source = ["SIDRA", "population_tensor"]
    provenance = ["official_sidra_anchor", "population_tensor", result.solver_id]
    warnings = list(result.warnings)
    return {
        "field_id": f"population_tensor_{result.mode}",
        "name": "PopulationTensorOfficialSIDRAIndependent" if result.mode == "independent_denominator" else "PopulationTensorSIMInformedWarningScaffold",
        "kind": "latent_context" if result.mode == "sim_informed_denominator" else "extensive_measure",
        "carrier": "Population",
        "unit": result.unit,
        "support_json": _compact(support),
        "axes_json": _compact(axes),
        "aggregation": "additive",
        "role": _compact(role),
        "role_json": _compact(role),
        "source": _compact(source),
        "source_json": _compact(source),
        "operator": "PopulationTensor/IndependentSIDRAAnchor" if result.mode == "independent_denominator" else "PopulationTensor/SIMInformedWarningScaffold",
        "provenance": _compact(provenance),
        "provenance_json": _compact(provenance),
        "state": result.state,
        "warnings": _compact(warnings),
        "warnings_json": _compact(warnings),
        "lineage_json": _compact({"parent_ids": [result.source_anchor_field_id], "operator": "population_tensor_solver", "created_at": _now(), "tensor_id": result.tensor_id}),
        "materialization_state": "materialized",
        "path": "Tables/population_tensor_diagnostics.parquet",
        "dashboard_safe": "warning" if result.warnings else "true",
    }
'''


Q_ROW_FUNCTION = r'''
def _q_row(field: dict[str, Any], result: PopulationTensorResult) -> dict[str, Any]:
    warnings = list(result.warnings)
    return {
        "field_id": field["field_id"],
        "n_events": result.value,
        "n_denom": result.value,
        "n_eff": result.value,
        "cov_S": 1.0,
        "cov_T": 1.0,
        "missingness": 0.0,
        "zero_inflation": 0.0,
        "denom_fragility": result.reconstruction_uncertainty,
        "cv": result.reconstruction_uncertainty,
        "moran_i": None,
        "temporal_roughness": None,
        "spatial_entropy": None,
        "provenance_risk": 0.05 if result.mode == "independent_denominator" else 0.35,
        "race_axis_source": None,
        "race_axis_target": None,
        "missing_race_share": None,
        "emission_prior_strength": None,
        "race_bridge_cv": None,
        "sensitivity_width": None,
        "bridge_mode": None,
        "state": result.state,
        "dashboard_safe": field["dashboard_safe"],
        "warnings": _compact(warnings),
        "warnings_json": _compact(warnings),
        "computed_at": _now(),
        "q_schema_version": "1.0",
    }
'''


def patch_population_tensor_bundle() -> None:
    p = path("src/pegasus/output/population_tensor_bundle.py")
    text = p.read_text(encoding="utf-8")
    text = replace_function(text, "_field", FIELD_FUNCTION)
    text = replace_function(text, "_q_row", Q_ROW_FUNCTION)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_validate() -> None:
    p = path("src/pegasus/output/validate.py")
    text = p.read_text(encoding="utf-8")
    old = '''        provenance = _load_json_cell(row.get("provenance_json"), errors=errors, context=f"V_fields.provenance_json[{fid}]")
        warnings_cell = _load_json_cell(row.get("warnings_json"), errors=errors, context=f"V_fields.warnings_json[{fid}]")'''
    new = '''        provenance_cell = row.get("provenance_json")
        if provenance_cell in (None, ""):
            provenance_cell = row.get("provenance")
        warnings_source_cell = row.get("warnings_json")
        if warnings_source_cell in (None, ""):
            warnings_source_cell = row.get("warnings")
        provenance = _load_json_cell(provenance_cell, errors=errors, context=f"V_fields.provenance[{fid}]")
        warnings_cell = _load_json_cell(warnings_source_cell, errors=errors, context=f"V_fields.warnings[{fid}]")'''
    if old in text:
        text = text.replace(old, new)
    elif "provenance_cell = row.get(\"provenance_json\")" not in text:
        raise RuntimeError("Cannot patch validate.py: population tensor provenance/warnings read block not found.")
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    write("src/pegasus/she/population/schema.py", POPULATION_SCHEMA)
    patch_population_tensor_bundle()
    write("src/pegasus/output/population_tensor_compile_attach.py", POPULATION_TENSOR_COMPILE_ATTACH)
    patch_validate()
    print("Applied Slice 6B population tensor contract repair v3.")
    print("Changed:")
    print("  src/pegasus/she/population/schema.py")
    print("  src/pegasus/output/population_tensor_bundle.py")
    print("  src/pegasus/output/population_tensor_compile_attach.py")
    print("  src/pegasus/output/validate.py")


if __name__ == "__main__":
    main()
