from typer.testing import CliRunner

from pegasus.cli import app


def test_datasus_ingest_dry_run_plans_manifest():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "datasus",
            "ingest",
            "--system",
            "SIM-DO",
            "--uf",
            "AL",
            "--years",
            "2022",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "planned" in result.output
    assert "SIM-DO" in result.output
