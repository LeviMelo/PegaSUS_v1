import json
from pathlib import Path

from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES, RunTelemetry


def test_run_telemetry_flushes_all_compile_stages(tmp_path: Path):
    diagnostic = tmp_path / "telemetry.json"
    telemetry = RunTelemetry(run_id="test_run", diagnostic_path=diagnostic)

    with telemetry.stage("datasus_manifest"):
        pass
    telemetry.block("pirs_model", reason="blocked in test")

    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    model = payload["telemetry"]

    assert set(COMPILE_TELEMETRY_STAGES).issubset(model["stage_status"])
    assert model["stage_status"]["datasus_manifest"] == "success"
    assert model["stage_status"]["pirs_model"] == "blocked"
    assert model["stage_wall_seconds"]["datasus_manifest"] >= 0
    assert model["total_wall_seconds"] >= 0
