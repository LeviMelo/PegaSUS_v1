from __future__ import annotations

import json
from pathlib import Path

from pegasus.core.schemas import UserIntent


def test_allsource_state_panel_intent_is_valid_user_intent() -> None:
    payload = json.loads(Path("config/intents/alagoas_2022_actual_allsource_state_panel.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)
    assert intent.geo_mode == "native"
    assert intent.execution_scale == "state"
    assert intent.geography.uf == ["AL"]
    assert intent.system_weights["CNES-ST"] == 1.0
    assert intent.system_weights["SIH-RD"] == 1.0
    assert "include_cnes_sih" in intent.context_policy


def test_actual_state_panel_runtime_filters_sidra_to_al_n6() -> None:
    source = Path("scripts/dev/audits/actual_state_panel_runtime.py").read_text(encoding="utf-8")
    assert "AL_N6_cod7_prefix_27" in source
    assert "expected 102 municipalities" in source
    assert ".startswith(\"27\")" in source
