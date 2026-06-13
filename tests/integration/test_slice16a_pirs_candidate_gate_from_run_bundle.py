
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
