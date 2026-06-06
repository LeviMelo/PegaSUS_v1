from __future__ import annotations

from pathlib import Path
import re
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def patch_cli() -> None:
    path = ROOT / "src" / "pegasus" / "cli.py"
    text = path.read_text(encoding="utf-8")

    if "attach_sidra_population_anchor_to_run" not in text:
        text = text.replace(
            "from pegasus.output.validate import validate_output_bundle\n",
            "from pegasus.output.validate import validate_output_bundle\n"
            "from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run\n",
            1,
        )

    command = '''@efg_app.command("attach-sidra-denominator")
def efg_attach_sidra_denominator(
    run_dir: Path = typer.Option(..., "--run-dir"),
    sidra_facts: Path = typer.Option(..., "--sidra-facts"),
) -> None:
    output = attach_sidra_population_anchor_to_run(
        run_dir=run_dir,
        sidra_facts_path=sidra_facts,
    )
    result = validate_output_bundle(run_dir=str(output))
    if not result.ok:
        _fail(result.errors)
    print(f"[green]SIDRA denominator anchor attached and run bundle valid[/green] {output}")
'''

    if '@efg_app.command("attach-sidra-denominator")' not in text:
        marker = '\n\n@sidra_app.command("metadata-fixture")'
        if marker not in text:
            raise RuntimeError("Could not find insertion point before sidra metadata-fixture command.")
        text = text.replace(marker, "\n\n" + command + marker, 1)

    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/she/population/sidra_anchor.py", r'''
    from __future__ import annotations

    import json
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash


    TOTAL_CATEGORY_SET_9606 = {
        ("2", "6794"),       # Sexo total
        ("86", "95251"),     # Cor ou raça total
        ("287", "100362"),   # Idade total
    }


    @dataclass(frozen=True)
    class SidraPopulationAnchor:
        field_id: str
        table_id: str
        variable_id: str
        period: str
        locality_level: str
        locality_id: str
        value: float
        unit: str
        request_hash: str
        metadata_hash: str
        classification_tuple: list[tuple[str, str]]
        category_tuple: list[tuple[str, str]]
        total_category_policy: str = "total_only"
        projection_operator: str = "Pi_Clsf_to_Axis/pi_bound_*"


    def _loads_tuple(value: Any) -> list[tuple[str, str]]:
        if value is None:
            return []
        if isinstance(value, str):
            parsed = json.loads(value)
        else:
            parsed = value
        return [tuple(map(str, x)) for x in parsed]


    def _is_total_9606(row: dict[str, Any]) -> bool:
        try:
            cats = set(_loads_tuple(row["category_tuple"]))
        except Exception:
            return False
        return TOTAL_CATEGORY_SET_9606 <= cats


    def load_sidra_population_total_anchor(facts_path: str | Path) -> SidraPopulationAnchor:
        facts_path = Path(facts_path)
        df = pl.read_parquet(facts_path)

        required = {
            "table_id",
            "variable_id",
            "period",
            "locality_level",
            "locality_id",
            "classification_tuple",
            "category_tuple",
            "value_numeric",
            "value_status",
            "unit",
            "request_hash",
            "metadata_hash",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"SIDRA facts file is missing required columns: {sorted(missing)}")

        candidates = []
        for row in df.to_dicts():
            if str(row["table_id"]) != "9606":
                continue
            if str(row["variable_id"]) != "93":
                continue
            if str(row["value_status"]) != "numeric":
                continue
            if row["value_numeric"] is None:
                continue
            if str(row.get("metadata_hash") or "") in {"", "metadata_unset"}:
                continue
            if not _is_total_9606(row):
                continue
            candidates.append(row)

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one total SIDRA 9606 population anchor fact; found {len(candidates)}."
            )

        row = candidates[0]
        category_tuple = _loads_tuple(row["category_tuple"])
        classification_tuple = _loads_tuple(row["classification_tuple"])

        field_payload = {
            "kind": "SIDRAPopulationTotalAnchor",
            "facts_path": str(facts_path),
            "table_id": str(row["table_id"]),
            "variable_id": str(row["variable_id"]),
            "period": str(row["period"]),
            "locality_level": str(row["locality_level"]),
            "locality_id": str(row["locality_id"]),
            "category_tuple": category_tuple,
            "request_hash": str(row["request_hash"]),
            "metadata_hash": str(row["metadata_hash"]),
        }

        return SidraPopulationAnchor(
            field_id=content_hash(field_payload),
            table_id=str(row["table_id"]),
            variable_id=str(row["variable_id"]),
            period=str(row["period"]),
            locality_level=str(row["locality_level"]),
            locality_id=str(row["locality_id"]),
            value=float(row["value_numeric"]),
            unit=str(row["unit"] or "Pessoas"),
            request_hash=str(row["request_hash"]),
            metadata_hash=str(row["metadata_hash"]),
            classification_tuple=classification_tuple,
            category_tuple=category_tuple,
        )
    ''')

    write("src/pegasus/output/sidra_denominator_anchor.py", r'''
    from __future__ import annotations

    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash
    from pegasus.output.validate import validate_output_bundle
    from pegasus.she.population.sidra_anchor import SidraPopulationAnchor, load_sidra_population_total_anchor


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


    def _load_json_field(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return json.loads(str(value))


    def _remove_by_values(df: pl.DataFrame, column: str, values: set[str]) -> pl.DataFrame:
        if column not in df.columns or not values:
            return df
        return df.filter(~pl.col(column).is_in(sorted(values)))


    def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str | None = None, remove_values: set[str] | None = None) -> None:
        df = pl.read_parquet(path)

        if remove_column and remove_values:
            df = _remove_by_values(df, remove_column, remove_values)

        if not rows:
            df.write_parquet(path)
            return

        add = pl.DataFrame(rows)

        for col, dtype in df.schema.items():
            if col not in add.columns:
                add = add.with_columns(pl.lit(None).cast(dtype).alias(col))
            else:
                add = add.with_columns(pl.col(col).cast(dtype, strict=False))

        add = add.select(df.columns)
        pl.concat([df, add], how="vertical").write_parquet(path)


    def _row_for_columns(columns: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        return {col: payload.get(col) for col in columns}


    def _get_field_by_name(v: pl.DataFrame, name: str) -> dict[str, Any]:
        rows = v.filter(pl.col("name") == name).to_dicts()
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one field named {name}; found {len(rows)}.")
        return rows[0]


    def _population_v_field(anchor: SidraPopulationAnchor, facts_path: Path) -> dict[str, Any]:
        support = {
            "support": "municipality_year",
            "years": [int(anchor.period) if anchor.period.isdigit() else anchor.period],
            "municipalities": [anchor.locality_id],
            "n_denom": anchor.value,
            "n_eff": anchor.value,
            "cov_S": 1.0,
            "cov_T": 1.0,
            "missingness": 0.0,
            "denom_fragility": 0.0,
            "sidra_table_id": anchor.table_id,
            "sidra_variable_id": anchor.variable_id,
            "total_category_policy": anchor.total_category_policy,
            "source_category_tuple": anchor.category_tuple,
            "source_classification_tuple": anchor.classification_tuple,
        }

        axes = {
            "time": "year",
            "geography": "municipality",
            "race_axis_type": None,
            "sidra_source_classifications": {
                "2": "sex",
                "86": "race_color",
                "287": "age",
            },
            "sidra_source_categories": dict(anchor.category_tuple),
            "projection_metadata": {
                "source_classifications": anchor.classification_tuple,
                "source_categories": anchor.category_tuple,
                "target_axes": ["municipality", "year"],
                "projection_matrix_id": "total_category_identity_marginal_9606_v1",
                "total_category_policy": "total_only",
                "fractional_mapping_warnings": [],
            },
            "bounded_pushforward": {
                "operator": "pi_bound_*",
                "axes_kept": ["municipality", "year"],
                "axes_dropped": ["sex", "race_color", "age"],
                "legal": True,
                "reason": "SIDRA 9606 total sex/race/age categories produce Population(s,t).",
            },
        }

        return {
            "field_id": anchor.field_id,
            "name": "SIDRAPopulationTotalAnchor",
            "kind": "extensive_measure",
            "carrier": "Population",
            "unit": "persons",
            "aggregation": "additive",
            "role": _json(["demographic", "exposure_offset"]),
            "source": _json(["SIDRA"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": "Pi_Clsf_to_Axis/pi_bound_*",
            "provenance": _json(["official", "sidra_9606", "bounded_total_category_anchor"]),
            "state": "fragile",
            "dashboard_safe": "warning",
            "warnings": _json(["sidra_total_category_anchor", "population_denominator_contract_fragile_until_crosscheck"]),
            "lineage_hash": anchor.field_id,
            "registry_hash": _json(
                {
                    "sidra_views": "v1.0",
                    "sidra_metadata_hash": anchor.metadata_hash,
                    "sidra_request_hash": anchor.request_hash,
                }
            ),
            "materialization_state": "metadata_only",
            "path": str(facts_path),
        }


    def _rate_v_field(*, all_deaths: dict[str, Any], anchor: SidraPopulationAnchor, facts_path: Path) -> dict[str, Any]:
        death_support = _load_json_field(all_deaths["support_json"])
        n_events = float(death_support.get("n_events", 0.0))

        field_id = content_hash(
            {
                "kind": "SIMCrudeMortalitySIDRAOfficial",
                "parents": [all_deaths["field_id"], anchor.field_id],
                "operator": "RN",
                "sidra_metadata_hash": anchor.metadata_hash,
                "sidra_request_hash": anchor.request_hash,
            }
        )

        support = {
            "support": "municipality_year",
            "years": [int(anchor.period) if anchor.period.isdigit() else anchor.period],
            "municipalities": [anchor.locality_id],
            "n_events": n_events,
            "n_denom": anchor.value,
            "n_eff": n_events,
            "cov_S": 1.0,
            "cov_T": 1.0,
            "missingness": float(death_support.get("missingness", 0.0)),
            "denom_fragility": 0.0,
            "sidra_population_anchor_field_id": anchor.field_id,
            "fixture_rate": True,
        }

        axes = {
            "time": "year",
            "geography": "municipality",
            "diagnostic_role": "all_deaths",
            "topology": "none",
            "race_axis_type": None,
            "denominator_source": "SIDRA_9606_total_population_anchor",
        }

        return {
            "field_id": field_id,
            "name": "SIMCrudeMortalitySIDRAOfficial",
            "kind": "intensive_density",
            "carrier": "Deaths/Population",
            "unit": "rate",
            "aggregation": "non_aggregable",
            "role": _json(["outcome", "model_only"]),
            "source": _json(["SIM-DO", "SIDRA"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": "RN",
            "provenance": _json(["official", "sidra_denominator_anchor", "sim_fixture_numerator"]),
            "state": "quarantined_descriptive",
            "dashboard_safe": "False",
            "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor", "dashboard_unsafe_fixture_rate"]),
            "lineage_hash": field_id,
            "registry_hash": _json(
                {
                    "sidra_views": "v1.0",
                    "sidra_metadata_hash": anchor.metadata_hash,
                    "sidra_request_hash": anchor.request_hash,
                }
            ),
            "materialization_state": "metadata_only",
            "path": str(facts_path),
        }


    def _q_rows(*, population_row: dict[str, Any], rate_row: dict[str, Any], anchor: SidraPopulationAnchor) -> list[dict[str, Any]]:
        pop_support = _load_json_field(population_row["support_json"])
        rate_support = _load_json_field(rate_row["support_json"])

        return [
            {
                "field_id": population_row["field_id"],
                "n_events": None,
                "n_denom": float(pop_support["n_denom"]),
                "n_eff": float(pop_support["n_eff"]),
                "cov_S": 1.0,
                "cov_T": 1.0,
                "missingness": 0.0,
                "zero_inflation": 0.0,
                "denom_fragility": 0.0,
                "cv": None,
                "moran_i": None,
                "temporal_roughness": None,
                "spatial_entropy": None,
                "provenance_risk": 0.0,
                "race_axis_source": None,
                "race_axis_target": None,
                "missing_race_share": None,
                "emission_prior_strength": None,
                "race_bridge_cv": None,
                "sensitivity_width": None,
                "bridge_mode": None,
                "state": "fragile",
                "dashboard_safe": "warning",
                "warnings": _json(["sidra_total_category_anchor"]),
                "computed_at": _now(),
                "q_schema_version": "1.0",
            },
            {
                "field_id": rate_row["field_id"],
                "n_events": float(rate_support["n_events"]),
                "n_denom": float(rate_support["n_denom"]),
                "n_eff": float(rate_support["n_eff"]),
                "cov_S": 1.0,
                "cov_T": 1.0,
                "missingness": float(rate_support["missingness"]),
                "zero_inflation": 0.0,
                "denom_fragility": 0.0,
                "cv": None,
                "moran_i": None,
                "temporal_roughness": None,
                "spatial_entropy": None,
                "provenance_risk": 0.2,
                "race_axis_source": None,
                "race_axis_target": None,
                "missing_race_share": None,
                "emission_prior_strength": None,
                "race_bridge_cv": None,
                "sensitivity_width": None,
                "bridge_mode": None,
                "state": "quarantined_descriptive",
                "dashboard_safe": "False",
                "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor"]),
                "computed_at": _now(),
                "q_schema_version": "1.0",
            },
        ]


    def _vd_rows(*, vd_columns: list[str], population_row: dict[str, Any], rate_row: dict[str, Any], anchor: SidraPopulationAnchor) -> list[dict[str, Any]]:
        pop_support = _load_json_field(population_row["support_json"])
        pop_axes = _load_json_field(population_row["axes_json"])
        rate_support = _load_json_field(rate_row["support_json"])
        rate_axes = _load_json_field(rate_row["axes_json"])

        return [
            _row_for_columns(
                vd_columns,
                {
                    "field_id": population_row["field_id"],
                    "display_name": "SIDRAPopulationTotalAnchor",
                    "technical_name": "SIDRAPopulationTotalAnchor",
                    "definition": "Official SIDRA 9606 total-category resident population anchor bounded to Population(s,t).",
                    "estimand_label": "official_resident_population_total_municipality_year",
                    "source_systems": _json(["SIDRA"]),
                    "carrier": "Population",
                    "unit": "persons",
                    "support_description": _json(pop_support),
                    "axis_description": _json(pop_axes),
                    "provenance_description": _json(["official", "sidra_9606", "bounded_total_category_anchor"]),
                    "state": "fragile",
                    "dashboard_safe": "warning",
                    "interpretation_warning": "Official SIDRA denominator anchor. Total sex/race/age categories are marginalization categories, not modeled axes.",
                    "diagnostic_role": None,
                    "topology": None,
                    "position": None,
                    "icd_group_kind": None,
                    "icd_group_id": None,
                },
            ),
            _row_for_columns(
                vd_columns,
                {
                    "field_id": rate_row["field_id"],
                    "display_name": "SIMCrudeMortalitySIDRAOfficial",
                    "technical_name": "SIMCrudeMortalitySIDRAOfficial",
                    "definition": "SIM fixture all-deaths numerator divided by official SIDRA total resident population denominator.",
                    "estimand_label": "fixture_crude_mortality_with_official_sidra_denominator",
                    "source_systems": _json(["SIM-DO", "SIDRA"]),
                    "carrier": "Deaths/Population",
                    "unit": "rate",
                    "support_description": _json(rate_support),
                    "axis_description": _json(rate_axes),
                    "provenance_description": _json(["official", "sidra_denominator_anchor", "sim_fixture_numerator"]),
                    "state": "quarantined_descriptive",
                    "dashboard_safe": "False",
                    "interpretation_warning": "Uses official SIDRA denominator but fixture SIM numerator; not a production analytic rate.",
                    "diagnostic_role": "all_deaths",
                    "topology": "none",
                    "position": None,
                    "icd_group_kind": None,
                    "icd_group_id": None,
                },
            ),
        ]


    def attach_sidra_population_anchor_to_run(
        *,
        run_dir: str | Path,
        sidra_facts_path: str | Path,
    ) -> Path:
        run_dir = Path(run_dir)
        sidra_facts_path = Path(sidra_facts_path)

        anchor = load_sidra_population_total_anchor(sidra_facts_path)

        v_path = run_dir / "V_fields.parquet"
        e_path = run_dir / "E_DAG.parquet"
        q_path = run_dir / "Q_tensor.parquet"
        vd_path = run_dir / "VariableDictionary.parquet"
        warnings_path = run_dir / "Warnings.parquet"
        qf_path = run_dir / "QuarantinedFields.parquet"
        p_path = run_dir / "P_vector.json"

        for path in [v_path, e_path, q_path, vd_path, warnings_path, qf_path, p_path]:
            if not path.exists():
                raise FileNotFoundError(f"Run bundle is missing required file: {path}")

        v = pl.read_parquet(v_path)
        all_deaths = _get_field_by_name(v, "SIMDeathsAll")

        population_row = _population_v_field(anchor, sidra_facts_path)
        rate_row = _rate_v_field(all_deaths=all_deaths, anchor=anchor, facts_path=sidra_facts_path)

        new_field_ids = {population_row["field_id"], rate_row["field_id"]}
        new_names = {"SIDRAPopulationTotalAnchor", "SIMCrudeMortalitySIDRAOfficial"}

        # V_fields
        v_clean = v.filter(~pl.col("name").is_in(sorted(new_names)))
        v_clean.write_parquet(v_path)
        _append_rows(v_path, [population_row, rate_row], remove_column="field_id", remove_values=new_field_ids)

        # E_DAG: only the new rate has parents.
        edge_rows = [
            {
                "edge_id": f"{all_deaths['field_id']}->{rate_row['field_id']}",
                "parent_field_id": all_deaths["field_id"],
                "child_field_id": rate_row["field_id"],
                "operator": "RN",
                "operator_params_json": _json({"role": "mortality_rate", "denominator": "SIDRA_9606_total_population_anchor"}),
                "registry_versions_json": _json({"sidra_metadata_hash": anchor.metadata_hash}),
                "created_at": _now(),
            },
            {
                "edge_id": f"{population_row['field_id']}->{rate_row['field_id']}",
                "parent_field_id": population_row["field_id"],
                "child_field_id": rate_row["field_id"],
                "operator": "RN",
                "operator_params_json": _json({"role": "mortality_rate", "denominator": "SIDRA_9606_total_population_anchor"}),
                "registry_versions_json": _json({"sidra_metadata_hash": anchor.metadata_hash}),
                "created_at": _now(),
            },
        ]
        e = pl.read_parquet(e_path)
        e = e.filter(pl.col("child_field_id") != rate_row["field_id"])
        e.write_parquet(e_path)
        _append_rows(e_path, edge_rows)

        # Q tensor
        _append_rows(q_path, _q_rows(population_row=population_row, rate_row=rate_row, anchor=anchor), remove_column="field_id", remove_values=new_field_ids)

        # Variable dictionary
        vd = pl.read_parquet(vd_path)
        vd_columns = vd.columns
        vd = vd.filter(~pl.col("field_id").is_in(sorted(new_field_ids)))
        vd.write_parquet(vd_path)
        _append_rows(vd_path, _vd_rows(vd_columns=vd_columns, population_row=population_row, rate_row=rate_row, anchor=anchor))

        # Warnings
        warning_rows = [
            {
                "warning_id": "sidra_total_category_anchor",
                "field_id": population_row["field_id"],
                "source": "pegasus.output.sidra_denominator_anchor",
                "severity": "info",
                "code": "sidra_total_category_anchor",
                "message": "SIDRA 9606 total sex/race/age categories were used to create bounded Population(s,t).",
                "inherited_from": _json([]),
                "created_at": _now(),
            },
            {
                "warning_id": "official_sidra_denominator_anchor",
                "field_id": rate_row["field_id"],
                "source": "pegasus.output.sidra_denominator_anchor",
                "severity": "info",
                "code": "official_sidra_denominator_anchor",
                "message": "SIM fixture crude mortality now has an official SIDRA total-population denominator variant.",
                "inherited_from": _json([population_row["field_id"]]),
                "created_at": _now(),
            },
        ]
        _append_rows(
            warnings_path,
            warning_rows,
            remove_column="warning_id",
            remove_values={row["warning_id"] for row in warning_rows},
        )

        # Quarantine metadata remains conservative because this is still a fixture numerator.
        qf_rows = [
            {
                "field_id": population_row["field_id"],
                "state": "fragile",
                "reason": "official_anchor_pending_crosschecks",
                "warnings": _json(["sidra_total_category_anchor"]),
            },
            {
                "field_id": rate_row["field_id"],
                "state": "quarantined_descriptive",
                "reason": "sim_fixture_numerator",
                "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor"]),
            },
        ]
        _append_rows(qf_path, qf_rows, remove_column="field_id", remove_values=new_field_ids)

        # P_vector provenance.
        p = json.loads(p_path.read_text(encoding="utf-8"))
        p.setdefault("provenance", {})
        p["provenance"][population_row["field_id"]] = ["official", "sidra_9606", "bounded_total_category_anchor"]
        p["provenance"][rate_row["field_id"]] = ["official", "sidra_denominator_anchor", "sim_fixture_numerator"]
        p_path.write_text(_json(p), encoding="utf-8")

        result = validate_output_bundle(run_dir=str(run_dir))
        if not result.ok:
            raise RuntimeError("Run bundle failed validation after SIDRA denominator anchor attach: " + "; ".join(result.errors))

        return run_dir
    ''')

    write("tests/unit/test_sidra_denominator_anchor.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run
    from pegasus.output.validate import validate_output_bundle
    from pegasus.sidra.facts import write_facts_parquet
    from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
    from pegasus.she.population.sidra_anchor import load_sidra_population_total_anchor
    from pegasus.workflows.build_efg import build_sim_fixture_efg_run


    LIVE_SHAPE_PAYLOAD = [
        {
            "NC": "Nível Territorial (Código)",
            "NN": "Nível Territorial",
            "MC": "Unidade de Medida (Código)",
            "MN": "Unidade de Medida",
            "V": "Valor",
            "D1C": "Município (Código)",
            "D1N": "Município",
            "D2C": "Ano (Código)",
            "D2N": "Ano",
            "D3C": "Variável (Código)",
            "D3N": "Variável",
            "D4C": "Sexo (Código)",
            "D4N": "Sexo",
            "D5C": "Cor ou raça (Código)",
            "D5N": "Cor ou raça",
            "D6C": "Idade (Código)",
            "D6N": "Idade",
        },
        {
            "NC": "6",
            "NN": "Município",
            "MC": "45",
            "MN": "Pessoas",
            "V": "957916",
            "D1C": "2704302",
            "D1N": "Maceió (AL)",
            "D2C": "2022",
            "D2N": "2022",
            "D3C": "93",
            "D3N": "População residente",
            "D4C": "6794",
            "D4N": "Total",
            "D5C": "95251",
            "D5N": "Total",
            "D6C": "100362",
            "D6N": "Total",
        },
    ]


    CHUNK_REQUEST = {
        "table_id": "9606",
        "variables": ["93"],
        "periods": ["2022"],
        "locality_level": "N6",
        "localities": ["2704302"],
        "classifications": {
            "86": ["95251"],
            "2": ["6794"],
            "287": ["100362"],
        },
    }


    def _sidra_facts(tmp_path: Path) -> Path:
        facts = normalize_sidra_payload_to_facts(
            LIVE_SHAPE_PAYLOAD,
            table_id="9606",
            request_hash="request_hash",
            metadata_hash="a" * 64,
            chunk_request=CHUNK_REQUEST,
            unit_by_variable=None,
            fetched_at="2026-06-06T00:00:00+00:00",
        )
        path = tmp_path / "sidra_facts.parquet"
        write_facts_parquet(facts, output_path=path)
        return path


    def test_total_population_anchor_loader(tmp_path: Path):
        facts_path = _sidra_facts(tmp_path)
        anchor = load_sidra_population_total_anchor(facts_path)

        assert anchor.table_id == "9606"
        assert anchor.variable_id == "93"
        assert anchor.period == "2022"
        assert anchor.locality_level == "N6"
        assert anchor.locality_id == "2704302"
        assert anchor.value == 957916.0
        assert anchor.metadata_hash == "a" * 64


    def test_attach_sidra_anchor_to_valid_run_bundle(tmp_path: Path):
        sim_events = tmp_path / "sim_events.parquet"
        run_dir = tmp_path / "run"
        sidra_facts = _sidra_facts(tmp_path)

        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=sim_events,
            source_manifest_hash="fixture_manifest_hash",
        )
        build_sim_fixture_efg_run(sim_events_path=sim_events, run_dir=run_dir)

        attach_sidra_population_anchor_to_run(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts,
        )

        result = validate_output_bundle(run_dir=str(run_dir))
        assert result.ok, result.errors

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        names = set(v["name"].to_list())

        assert "SIDRAPopulationTotalAnchor" in names
        assert "SIMCrudeMortalitySIDRAOfficial" in names

        pop = v.filter(pl.col("name") == "SIDRAPopulationTotalAnchor").row(0, named=True)
        rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)

        assert pop["carrier"] == "Population"
        assert pop["source"] == '["SIDRA"]'
        assert "bounded_total_category_anchor" in pop["provenance"]

        assert rate["carrier"] == "Deaths/Population"
        assert "SIDRA" in rate["source"]

        e = pl.read_parquet(run_dir / "E_DAG.parquet")
        assert rate["field_id"] in set(e["child_field_id"].to_list())

        q = pl.read_parquet(run_dir / "Q_tensor.parquet")
        assert pop["field_id"] in set(q["field_id"].to_list())
        assert rate["field_id"] in set(q["field_id"].to_list())
    ''')

    patch_cli()
    print("Applied Slice 2C: official SIDRA 9606 total-population denominator anchor and EFG-safe field attach workflow.")


if __name__ == "__main__":
    main()