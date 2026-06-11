import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


def test_compile_smoke_integrates_sinasc_birth_and_mortality_fields(tmp_path: Path):
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=tmp_path / "compile_run",
        data_root=tmp_path / "data",
    )
    run_dir = Path(result["run_dir"])
    validation = validate_output_bundle(run_dir=str(run_dir))
    assert validation.ok, validation.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    field_ids = set(v["field_id"].to_list())
    required = {
        "SINASCLiveBirthsAll",
        "SINASCCrudeBirthRateSIDRAOfficial",
        "SINASCLowBirthWeightPrevalence",
        "SINASCPrematurityPrevalence",
        "SINASCCongenitalAnomalyPrevalence",
        "SIMInfantMortalitySINASCBirths",
        "SIMNeonatalMortalitySINASCBirths",
        "SIMPostNeonatalMortalitySINASCBirths",
    }
    assert required.issubset(field_ids)

    table = pl.read_parquet(run_dir / "Tables" / "maternal_child_linkage_summary.parquet")
    assert table["births_total"].item() == 3
    assert table["denominator_population"].item() == 957916

    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["maternal_child_linkage"]["enabled"] is True
    assert "sinasc_processed_events" in run_config["source_hashes"]

    q = pl.read_parquet(run_dir / "Q_tensor.parquet")
    q_ids = set(q["field_id"].to_list())
    assert required.issubset(q_ids)
