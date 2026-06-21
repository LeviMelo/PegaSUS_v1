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
