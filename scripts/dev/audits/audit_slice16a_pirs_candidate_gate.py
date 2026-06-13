
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
