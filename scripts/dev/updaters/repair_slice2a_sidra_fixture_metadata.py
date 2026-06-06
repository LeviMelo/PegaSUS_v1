from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/sidra/metadata.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import polars as pl

    from pegasus.sidra.schemas import SIDRAMetadata, SIDRATableMetadata


    def load_table_seed(path: str | Path) -> list[dict]:
        rows: list[dict] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows


    def fixture_sidra_metadata() -> SIDRAMetadata:
        # Fixture metadata is deliberately broader than the tiny value fixture.
        # It exists to test metadata validation, deterministic chunk splitting,
        # and long-form SIDRA fact normalization without performing live HTTP.
        table = SIDRATableMetadata(
            table_id="9606",
            name="Population by municipality, period and classification fixture",
            variables=["93"],
            periods=["2022", "2023"],
            locality_levels=["N6"],
            localities_by_level={
                "N6": ["270030", "270430", "270770"],
            },
            classifications={
                # 2 = sexo in common SIDRA layouts. Keep total + binary categories
                # in the fixture so large planner tests can validate normally.
                "2": ["0", "1", "2"],

                # 58 = cor/raça in common SIDRA layouts. Include total and the
                # usual category range used by population tables.
                "58": ["0", "1", "2", "3", "4", "5", "9"],

                # 287 = idade/age grouping in the fixture. This remains partial;
                # tests may extend it explicitly when stress-testing chunking.
                "287": ["0", "93070", "93084", "100000"],
            },
            units_by_variable={"93": "persons"},
        )
        return SIDRAMetadata(tables={"9606": table})


    def write_normalized_metadata_tables(
        metadata: SIDRAMetadata,
        *,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = []
        variables = []
        classifications = []
        categories = []
        periods = []
        localities = []

        for table in metadata.tables.values():
            tables.append({"table_id": table.table_id, "name": table.name})

            for variable in table.variables:
                variables.append(
                    {
                        "table_id": table.table_id,
                        "variable_id": variable,
                        "unit": table.units_by_variable.get(variable),
                    }
                )

            for period in table.periods:
                periods.append({"table_id": table.table_id, "period": period})

            for level, locs in table.localities_by_level.items():
                for loc in locs:
                    localities.append(
                        {
                            "table_id": table.table_id,
                            "locality_level": level,
                            "locality_id": loc,
                        }
                    )

            for cls_id, cats in table.classifications.items():
                classifications.append(
                    {
                        "table_id": table.table_id,
                        "classification_id": cls_id,
                    }
                )
                for cat in cats:
                    categories.append(
                        {
                            "table_id": table.table_id,
                            "classification_id": cls_id,
                            "category_id": cat,
                        }
                    )

        outputs = {
            "sidra_tables": output_dir / "sidra_tables.parquet",
            "sidra_variables": output_dir / "sidra_variables.parquet",
            "sidra_classifications": output_dir / "sidra_classifications.parquet",
            "sidra_categories": output_dir / "sidra_categories.parquet",
            "sidra_periods": output_dir / "sidra_periods.parquet",
            "sidra_localities": output_dir / "sidra_localities.parquet",
        }

        pl.DataFrame(tables).write_parquet(outputs["sidra_tables"])
        pl.DataFrame(variables).write_parquet(outputs["sidra_variables"])
        pl.DataFrame(classifications).write_parquet(outputs["sidra_classifications"])
        pl.DataFrame(categories).write_parquet(outputs["sidra_categories"])
        pl.DataFrame(periods).write_parquet(outputs["sidra_periods"])
        pl.DataFrame(localities).write_parquet(outputs["sidra_localities"])

        return outputs
    ''')

    print("Repaired Slice 2A SIDRA fixture metadata category sets.")


if __name__ == "__main__":
    main()