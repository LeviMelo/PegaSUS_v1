from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def patch_cli() -> None:
    path = ROOT / "src" / "pegasus" / "cli.py"
    text = path.read_text(encoding="utf-8")

    import_line = "from pegasus.datasus.normalize import normalize_sim_do_events\n"

    if import_line not in text:
        if "from pegasus.datasus.manifests import (" in text:
            text = text.replace(
                "from pegasus.datasus.manifests import (\n",
                import_line + "from pegasus.datasus.manifests import (\n",
                1,
            )
        elif "from pegasus.datasus.profile import profile_table" in text:
            text = text.replace(
                "from pegasus.datasus.profile import profile_table\n",
                import_line + "from pegasus.datasus.profile import profile_table\n",
                1,
            )
        else:
            raise RuntimeError("Could not find a stable import insertion point in cli.py.")

    command = '''

@datasus_app.command("normalize-sim")
def datasus_normalize_sim(
    input_path: Path = typer.Option(..., "--input"),
    output_path: Path = typer.Option(..., "--output"),
    source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
) -> None:
    result = normalize_sim_do_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )
    print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")
'''

    if '@datasus_app.command("normalize-sim")' not in text:
        markers = [
            '@sidra_app.command("metadata")',
            '@app.command()\\ndef compile',
        ]

        inserted = False
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx] + command + "\n" + text[idx:]
                inserted = True
                break

        if not inserted:
            text = text.rstrip() + "\n" + command + "\n"

    path.write_text(text, encoding="utf-8", newline="\n")


def patch_tests() -> None:
    write("tests/unit/test_sim_normalize.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.datasus.normalize import SIM_DO_NORMALIZED_COLUMNS, normalize_sim_do_events


    def test_sim_do_fixture_normalization_schema(tmp_path: Path):
        output = tmp_path / "sim_events.parquet"

        result = normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=output,
            source_manifest_hash="fixture_manifest_hash",
        )

        assert result["row_count"] == 3
        assert output.exists()

        df = pl.read_parquet(output)
        assert df.columns == SIM_DO_NORMALIZED_COLUMNS
        assert df.height == 3


    def test_sim_do_age_and_icd_topology_normalization(tmp_path: Path):
        output = tmp_path / "sim_events.parquet"

        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=output,
            source_manifest_hash="fixture_manifest_hash",
        )

        df = pl.read_parquet(output).sort("death_date")

        first = df.row(0, named=True)

        # Date-derived age is preferred when DTNASC and DTOBITO are both valid.
        # The raw SIM structural age code is still preserved as provenance.
        assert first["age_source"] == "date_difference"
        assert abs(first["age_years"] - 74.0) < 0.01
        assert first["raw_age_code"] == "474"
        assert first["age_unit"] == "years"

        assert first["underlying_icd_norm"] == "A419"
        assert first["underlying_icd_parse_state"] == "valid"
        assert '"A": "*J189"' in first["cause_chain_raw"]
        assert '"A": "J189"' in first["cause_chain_norm"]

        second = df.row(1, named=True)
        assert second["age_source"] == "IDADE"
        assert second["underlying_icd_norm"] == "R99"
        assert second["underlying_icd_parse_state"] == "ill-defined"
        assert second["race_missingness_state"] == "unknown"


    def test_sim_do_missing_and_sentinel_values_are_not_silent_negatives(tmp_path: Path):
        output = tmp_path / "sim_events.parquet"

        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=output,
            source_manifest_hash="fixture_manifest_hash",
        )

        df = pl.read_parquet(output).sort("death_date")

        second = df.row(1, named=True)
        assert second["maternal_living_children_count"] is None
        assert second["birth_weight_death_context_grams"] is None
        assert second["facility_code"] is None
        assert second["facility_code_state"] == "missing"

        third = df.row(2, named=True)
        assert third["death_hour"] is None
        assert third["associated_conditions_norm"] == "Q249"
        assert third["associated_conditions_parse_states"] == "valid"
    ''')


def main() -> None:
    patch_cli()
    patch_tests()
    print("Repaired Slice 1C CLI insertion and age assertion.")


if __name__ == "__main__":
    main()