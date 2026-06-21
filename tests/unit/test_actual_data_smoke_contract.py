from __future__ import annotations

import importlib.util
from pathlib import Path


def _runtime_module():
    path = Path("scripts/dev/audits/actual_data_smoke_runtime.py")
    spec = importlib.util.spec_from_file_location("actual_data_smoke_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_smoke_never_classifies_missing_r_dependencies_as_success(monkeypatch) -> None:
    module = _runtime_module()
    monkeypatch.setattr(module, "_find_rscript", lambda: None)
    payload = module.run_actual_smoke(intent_name="alagoas_maceio_2022_actual_smoke.json", grid=False)
    assert payload["classification"] == "source_unavailable"
    assert payload["compile_attempted"] is False
    assert payload["compile_source_mode"] is None
    assert payload["run_dir"] is None


def test_actual_smoke_classifies_clean_materialized_bundle_as_validated() -> None:
    module = _runtime_module()
    payload = {
        "compile_source_mode": "materialized_external",
        "output_validator_ok": True,
        "dashboard_read_only_ok": True,
        "mandatory_fields_present": True,
        "efg_fields_nonempty": True,
        "efg_edges_nonempty": True,
        "q_tensor_nonempty": True,
        "fixture_semantics_present": [],
        "municipality_count": 1,
    }
    assert module.classify_actual_smoke(payload, grid=False) == "actual_data_validated"
    assert module.classify_actual_smoke(payload, grid=True) == "source_partial"
