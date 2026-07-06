from __future__ import annotations

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
