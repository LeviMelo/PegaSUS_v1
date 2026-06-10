import json
from pathlib import Path

from pegasus.core.schemas import UserIntent


def test_alagoas_smoke_intent_is_strict_user_intent():
    payload = json.loads(Path("config/intents/alagoas_smoke.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)

    assert intent.execution_scale == "smoke"
    assert intent.population_mode == "official_sidra_anchor"
    assert intent.geography.level == "municipality"
    assert intent.geography.codes == ["2704302"]
    assert intent.budget == "fast"
