from pathlib import Path

from typer.testing import CliRunner

from pegasus.cli import app


def test_slice3a_cli_commands_delegate_to_workflows():
    cli = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
    assert "run_datasus_normalize_sinasc" in cli
    assert "run_build_sinasc_fixture" in cli
    assert '@datasus_app.command("normalize-sinasc")' in cli
    assert '@efg_app.command("build-sinasc-fixture")' in cli
    assert "normalize_sinasc_events(" not in cli
    assert "write_sinasc_fixture_efg_bundle(" not in cli


def test_slice3a_cli_normalize_and_build_commands(tmp_path: Path):
    runner = CliRunner()
    events = tmp_path / "sinasc_events.parquet"
    result = runner.invoke(
        app,
        [
            "datasus", "normalize-sinasc",
            "--input", "tests/fixtures/datasus/sinasc_fixture.csv",
            "--output", str(events),
            "--source-manifest-hash", "fixture_manifest",
        ],
    )
    assert result.exit_code == 0, result.output
    assert events.exists()

    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "efg", "build-sinasc-fixture",
            "--sinasc-events", str(events),
            "--run-dir", str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "V_fields.parquet").exists()
