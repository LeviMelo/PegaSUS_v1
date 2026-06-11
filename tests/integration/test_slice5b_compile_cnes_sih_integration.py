from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from pegasus.core.schemas import UserIntent
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


def _field_ids(run: Path) -> set[str]:
    return {str(row["field_id"]) for row in pq.read_table(run / "V_fields.parquet").to_pylist()}


def _row(run: Path, field_id: str) -> dict:
    rows = pq.read_table(run / "V_fields.parquet").to_pylist()
    matches = [row for row in rows if row["field_id"] == field_id]
    assert matches, field_id
    return matches[0]


def test_cnes_sih_compile_intent_is_strict_user_intent():
    payload = json.loads(Path("config/intents/alagoas_smoke_cnes_sih.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)
    assert intent.context_policy == ["include_cnes_sih"]
    assert intent.race_tensor_mode == "decoupled"


def test_baseline_compile_remains_without_cnes_sih_fields(tmp_path: Path):
    run = tmp_path / "baseline"
    result = run_compile(intent_path="config/intents/alagoas_smoke.json", run_dir=run)
    assert result["validation"].ok, result["validation"].errors
    ids = _field_ids(run)
    assert not any(field_id.startswith("cnes_") for field_id in ids)
    assert not any(field_id.startswith("sih_cost_") for field_id in ids)


def test_compile_integrates_cnes_capacity_and_sih_cost_components(tmp_path: Path):
    run = tmp_path / "cnes_sih"
    result = run_compile(intent_path="config/intents/alagoas_smoke_cnes_sih.json", run_dir=run)
    assert result["validation"].ok, result["validation"].errors
    ids = _field_ids(run)

    assert "cnes_facilities_all" in ids
    assert "cnes_capacity_qtinst34" in ids
    assert "cnes_zero_facility_cnpj_share" in ids
    assert "sih_admissions_all" in ids
    assert "sih_inpatient_fatality" in ids
    assert "sih_cost_val_sh" in ids
    assert "sih_cost_val_sp" in ids
    assert "sih_cost_val_uti" in ids
    assert "sih_cost_val_tot" in ids

    cnes_axes = json.loads(_row(run, "cnes_capacity_qtinst34")["axes_json"])
    assert cnes_axes["capacity_vector_index"] == "QTINST34"
    assert cnes_axes["generic_beds_forbidden"] is True

    sih_axes = json.loads(_row(run, "sih_cost_val_tot")["axes_json"])
    assert sih_axes["cost_component"] == "VAL_TOT"
    assert sih_axes["generic_sih_cost_forbidden"] is True

    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    assert run_config["cnes_sih"]["cnes"]["generic_beds_blocked"] is True
    assert run_config["cnes_sih"]["sih"]["generic_sih_cost_blocked"] is True
    assert manifest["cnes_sih"]["sih"]["diagnostic_topology_preserved"] is True
    assert "cnes_processed_events" in run_config["source_hashes"]
    assert "sih_processed_events" in run_config["source_hashes"]
    assert (run / "Tables" / "slice5a_cnes_capacity_summary.parquet").exists()
    assert (run / "Tables" / "slice5a_sih_cost_summary.parquet").exists()


def test_validator_rejects_cnes_sih_fields_without_metadata(tmp_path: Path):
    run = tmp_path / "cnes_sih_invalid"
    result = run_compile(intent_path="config/intents/alagoas_smoke_cnes_sih.json", run_dir=run)
    assert result["validation"].ok, result["validation"].errors

    run_config_path = run / "RunConfig.json"
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    payload.pop("cnes_sih")
    run_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    validation = validate_output_bundle(run_dir=str(run))
    assert not validation.ok
    assert any("missing cnes_sih metadata" in error for error in validation.errors)


def test_validator_rejects_erased_capacity_vector_index(tmp_path: Path):
    run = tmp_path / "cnes_sih_bad_axis"
    result = run_compile(intent_path="config/intents/alagoas_smoke_cnes_sih.json", run_dir=run)
    assert result["validation"].ok, result["validation"].errors

    table = pq.read_table(run / "V_fields.parquet")
    rows = table.to_pylist()
    for row in rows:
        if row["field_id"] == "cnes_capacity_qtinst34":
            axes = json.loads(row["axes_json"])
            axes.pop("capacity_vector_index")
            row["axes_json"] = json.dumps(axes)
    pq.write_table(type(table).from_pylist(rows, schema=table.schema), run / "V_fields.parquet")

    validation = validate_output_bundle(run_dir=str(run))
    assert not validation.ok
    assert any("CNES capacity field missing axes metadata" in error for error in validation.errors)
