import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


REQUIRED_SUCCESS_STAGES = {
    "datasus_manifest",
    "datasus_acquire",
    "datasus_decode",
    "sidra_normalize",
    "efg_build",
    "geo_support",
    "she_build",
    "q_tensor",
    "output_serialization",
    "output_validation",
}


REQUIRED_BLOCKED_STAGES = {"population_solver", "stdfm", "pirs_model", "pirs_hsic"}


def test_compile_smoke_emits_valid_17_key_bundle_with_sidra_denominator(tmp_path: Path):
    run_dir = tmp_path / "compile_run"
    data_root = tmp_path / "data"

    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
        data_root=data_root,
    )

    assert result["status"] == "success"
    assert result["validation"].ok, result["validation"].errors

    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors

    intent = json.loads((run_dir / "UserIntent.json").read_text(encoding="utf-8"))
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

    assert intent["geography"]["codes"] == ["2704302"]
    assert run_config["compile_mode"] == "smoke"
    assert run_config["support_policy"]["datasus_cod6"] == ["270430"]
    assert "source_hashes" in run_config
    assert "registry_hashes" in run_config

    telemetry = manifest["telemetry"]
    for stage in REQUIRED_SUCCESS_STAGES:
        assert telemetry["stage_status"][stage] == "success", stage
        assert telemetry["stage_wall_seconds"][stage] >= 0, stage
    for stage in REQUIRED_BLOCKED_STAGES:
        assert telemetry["stage_status"][stage] == "blocked", stage

    for key in ["intent", "compile_manifest", "sim_raw_fixture", "sim_processed_events", "sidra_facts"]:
        assert key in manifest["source_hashes"]

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    names = set(v["name"].to_list())

    assert "SIMDeathsAll" in names
    assert "SIDRAPopulationTotalAnchor" in names
    assert "SIMCrudeMortalitySIDRAOfficial" in names

    rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)
    support = json.loads(rate["support_json"])
    assert support["support_alignment"]["aligned"] is True
    assert support["support_alignment"]["numerator_municipalities_ibge_cod7"] == ["2704302"]
    assert support["support_alignment"]["denominator_municipalities_ibge_cod7"] == ["2704302"]
