from pathlib import Path

import polars as pl

from pegasus.datasus.normalize import normalize_sim_do_events
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.build_efg import build_sim_fixture_efg_run


def _build(tmp_path: Path) -> Path:
    sim_events = tmp_path / "sim_events.parquet"
    run_dir = tmp_path / "run"

    normalize_sim_do_events(
        input_path="tests/fixtures/datasus/sim_do_fixture.csv",
        output_path=sim_events,
        source_manifest_hash="fixture_manifest_hash",
    )

    build_sim_fixture_efg_run(sim_events_path=sim_events, run_dir=run_dir)
    return run_dir


def test_icd_chapter_block_and_crude_mortality_nodes_exist(tmp_path: Path):
    run_dir = _build(tmp_path)

    result = validate_output_bundle(run_dir=str(run_dir))
    assert result.ok, result.errors

    v = pl.read_parquet(run_dir / "V_fields.parquet")
    names = set(v["name"].to_list())

    assert "SIMCrudeMortalityFixture" in names
    assert "SIMUnderlyingICDChapterDeaths" in names
    assert "SIMUnderlyingICDChapterMortality" in names
    assert "SIMUnderlyingICDBlockDeaths" in names
    assert "SIMUnderlyingICDBlockMortality" in names


def test_diagnostic_topology_is_not_collapsed(tmp_path: Path):
    run_dir = _build(tmp_path)

    vd = pl.read_parquet(run_dir / "VariableDictionary.parquet")

    chapter = vd.filter(pl.col("display_name") == "SIMUnderlyingICDChapterMortality").row(0, named=True)
    chain = vd.filter(pl.col("display_name") == "SIMTerminalChainMentionShare").row(0, named=True)
    associated = vd.filter(pl.col("display_name") == "SIMAssociatedConditionMentionShare").row(0, named=True)

    assert chapter["diagnostic_role"] == "underlying_cause"
    assert chapter["topology"] == "single_underlying"
    assert chapter["icd_group_kind"] == "chapter"

    assert chain["diagnostic_role"] == "terminal_chain"
    assert chain["topology"] == "ordered_terminal_chain"
    assert chain["estimand_label"] == "terminal_chain_mention_observer_share"
    assert "not cause-specific mortality" in chain["interpretation_warning"].lower()

    assert associated["diagnostic_role"] == "associated_condition"
    assert associated["topology"] == "unordered_associated_set"
    assert associated["estimand_label"] == "associated_condition_mention_observer_share"
    assert "not cause-specific mortality" in associated["interpretation_warning"].lower()


def test_failed_race_declaration_branch_still_exists(tmp_path: Path):
    run_dir = _build(tmp_path)

    failed = pl.read_parquet(run_dir / "FailedBranches.parquet")
    assert failed.height == 1
    row = failed.row(0, named=True)
    assert row["failure_stage"] == "declaration"
    assert "Bridge_R" in row["reason"]
