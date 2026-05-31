from __future__ import annotations

from typing import Literal

import polars as pl

from pegasus.problem1.compiled import CompiledField
from pegasus.problem1.contracts import FieldNode, Lineage, QState, WarningRecord


def _first_non_null_string(df: pl.DataFrame, column: str, fallback: str) -> str:
    if column not in df.columns or df.is_empty():
        return fallback

    values = df.select(pl.col(column).drop_nulls().cast(pl.Utf8)).to_series().to_list()
    return str(values[0]) if values else fallback


def _unique_non_null_strings(df: pl.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if column not in df.columns or df.is_empty():
        return fallback

    values = (
        df.select(pl.col(column).drop_nulls().cast(pl.Utf8).unique())
        .to_series()
        .to_list()
    )

    values = sorted({str(value) for value in values if value is not None})
    return values if values else fallback


def _provenance_risk(provenance: list[str]) -> float:
    risk_map = {
        "official": 0.0,
        "harmonized": 0.2,
        "deflated": 0.2,
        "AMC_contracted": 0.2,
        "imported_fixture": 0.2,
        "reconstructed": 0.4,
        "geneallocated": 0.4,
        "SIM_informed_denominator_prior": 0.5,
        "latent": 0.8,
        "synthetic": 1.0,
    }

    if not provenance:
        return 0.4

    return max(risk_map.get(item, 0.4) for item in provenance)


def compile_imported_population_field(
    population: pl.DataFrame,
    *,
    registry_hashes: dict[str, str],
    source_label: str = "imported_population_fixture",
    geography: Literal["municipality"] = "municipality",
) -> CompiledField:
    """Compile a population/person-years denominator field.

    Expected input columns:

        year
        municipality_cod6
        population

    Optional columns:

        population_state
        denominator_source
        provenance
        municipality_cod7
        municipality_name
        sidra_measure_id
    """
    required = {"year", "municipality_cod6", "population"}
    missing = sorted(required - set(population.columns))

    if missing:
        raise ValueError(f"Population table missing required columns: {missing}")

    df = population.select(
        [
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("municipality_cod6").cast(pl.Utf8, strict=False),
            pl.col("population").cast(pl.Float64, strict=False),
            (
                pl.col("population_state").cast(pl.Utf8, strict=False)
                if "population_state" in population.columns
                else pl.lit("valid").alias("population_state")
            ),
            (
                pl.col("denominator_source").cast(pl.Utf8, strict=False)
                if "denominator_source" in population.columns
                else pl.lit(source_label).alias("denominator_source")
            ),
            (
                pl.col("provenance").cast(pl.Utf8, strict=False)
                if "provenance" in population.columns
                else pl.lit("imported_fixture").alias("provenance")
            ),
        ]
    )

    invalid = df.filter(
        pl.col("year").is_null()
        | pl.col("municipality_cod6").is_null()
        | pl.col("population").is_null()
        | (pl.col("population") <= 0)
        | (pl.col("population_state") != "valid")
    )

    valid = df.filter(
        pl.col("year").is_not_null()
        & pl.col("municipality_cod6").is_not_null()
        & pl.col("population").is_not_null()
        & (pl.col("population") > 0)
        & (pl.col("population_state") == "valid")
    )

    denominator_source = _first_non_null_string(
        valid,
        "denominator_source",
        source_label,
    )
    provenance = _unique_non_null_strings(
        valid,
        "provenance",
        ["imported_fixture"],
    )

    valid = (
        valid.group_by(["year", "municipality_cod6"])
        .agg(
            pl.col("population").sum().alias("population"),
            pl.col("denominator_source").first().alias("denominator_source"),
            pl.col("provenance").first().alias("provenance"),
        )
        .with_columns(
            pl.lit("Population").alias("carrier"),
            pl.lit("person_years").alias("unit"),
            pl.lit(geography).alias("geography_basis"),
        )
        .select(
            [
                "year",
                "municipality_cod6",
                "population",
                "carrier",
                "unit",
                "geography_basis",
                "denominator_source",
                "provenance",
            ]
        )
        .sort(["year", "municipality_cod6"])
    )

    warnings: list[WarningRecord] = []
    missingness = invalid.height / df.height if df.height else 1.0

    if invalid.height:
        warnings.append(
            WarningRecord(
                field_id=None,
                source="compile_imported_population_field",
                severity="warning" if missingness < 0.20 else "severe",
                code="population_rows_excluded",
                message=(
                    f"{invalid.height}/{df.height} population rows excluded because "
                    "year, municipality, population, or population_state was invalid."
                ),
            )
        )

    state = "verified" if valid.height > 0 and missingness < 0.05 else "fragile"
    dashboard_safe: bool | str = True if state == "verified" else "warning"

    source_systems = ["IBGE", denominator_source]

    lineage = Lineage(
        parent_ids=[],
        operator_type="import_population_denominator",
        operator_params={
            "source_label": denominator_source,
            "geography": geography,
            "time_axis": "year",
            "carrier": "Population",
            "unit": "person_years",
            "provenance": provenance,
        },
        registry_versions=registry_hashes,
    )

    field = FieldNode.from_lineage(
        name=f"{denominator_source} population denominator by municipality and year",
        kind="extensive_measure",
        carrier="Population",
        unit="person_years",
        support={
            "time": "year",
            "geography": "municipality",
            "spatial_code": "municipality_cod6",
        },
        axes={
            "T": "year",
            "S*": "municipality",
        },
        aggregation="additive",
        role=["exposure_offset", "denominator", "population"],
        source=source_systems,
        operator="import_population_denominator",
        provenance=provenance,
        state=state,
        dashboard_safe=dashboard_safe,
        warnings=[warning.code for warning in warnings],
        lineage=lineage,
        materialization_state="materialized",
    )

    warnings = [warning.model_copy(update={"field_id": field.id}) for warning in warnings]

    q_state = QState(
        field_id=field.id,
        n_events=None,
        n_denom=float(valid["population"].sum()) if valid.height else None,
        n_eff=float(valid.height),
        cov_S=None,
        cov_T=None,
        missingness=missingness,
        zero_inflation=None,
        denom_fragility=missingness,
        cv=None,
        moran_i=None,
        temporal_roughness=None,
        spatial_entropy=None,
        provenance_risk=_provenance_risk(provenance),
        race_bridge_cv=None,
        sensitivity_width=None,
        state=state,
        dashboard_safe=dashboard_safe,
        warnings=[warning.code for warning in warnings],
    )

    return CompiledField(
        field=field,
        data=valid,
        q_state=q_state,
        warnings=warnings,
    )