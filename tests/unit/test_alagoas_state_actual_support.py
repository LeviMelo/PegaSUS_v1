from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pegasus.core.schemas import UserIntent
from pegasus.workflows.compile import _intent_municipality_filter_cod6, _smoke_municipality_cod6


def test_compile_support_resolver_allows_alagoas_state_without_municipality_filter() -> None:
    payload = json.loads(Path("config/intents/alagoas_2022_actual_grid_smoke.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)

    assert intent.execution_scale == "state"
    assert intent.geography.uf == ["AL"]
    assert intent.geography.codes == []
    assert _intent_municipality_filter_cod6(intent) is None


def test_compile_support_resolver_preserves_maceio_smoke_filter() -> None:
    payload = json.loads(Path("config/intents/alagoas_maceio_2022_actual_smoke.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)

    assert _intent_municipality_filter_cod6(intent) == "270430"
    assert _smoke_municipality_cod6(intent) == "270430"


def test_actual_smoke_grid_classifier_requires_multi_municipality_support() -> None:
    path = Path("scripts/dev/audits/actual_data_smoke_runtime.py")
    spec = importlib.util.spec_from_file_location("actual_data_smoke_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {
        "compile_source_mode": "materialized_external",
        "output_validator_ok": True,
        "dashboard_read_only_ok": True,
        "mandatory_fields_present": True,
        "efg_fields_nonempty": True,
        "efg_edges_nonempty": True,
        "q_tensor_nonempty": True,
        "fixture_semantics_present": [],
        "municipality_count": 102,
    }

    assert module.classify_actual_smoke(payload, grid=True) == "actual_data_validated"

    payload["municipality_count"] = 1
    assert module.classify_actual_smoke(payload, grid=True) == "source_partial"
