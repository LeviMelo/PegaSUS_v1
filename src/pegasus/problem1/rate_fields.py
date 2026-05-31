from __future__ import annotations

import polars as pl

from pegasus.problem1.compiled import CompiledField
from pegasus.problem1.contracts import FieldNode, Lineage, QState, WarningRecord
from pegasus.problem1.legality import RateLegalityRequest, evaluate_rate_legality
from pegasus.registries.index import RegistryIndex


def compile_crude_mortality_rate_field(
    *,
    death_count_field: FieldNode,
    death_count_data: pl.DataFrame,
    population_field: FieldNode,
    population_data: pl.DataFrame,
    registry_index: RegistryIndex,
    registry_hashes: dict[str, str],
    scale: float = 100_000.0,
) -> CompiledField:
    """Compile crude mortality rate from Deaths and Population fields.

    This is deliberately generic at the field level: it does not know SIM raw
    columns. It only depends on a Deaths/counts numerator and a
    Population/person-years denominator.
    """
    legality = evaluate_rate_legality(
        RateLegalityRequest(
            numerator_carrier=death_count_field.carrier,
            denominator_carrier=population_field.carrier,
            role="mortality_rate",
            numerator_unit=death_count_field.unit,
            denominator_unit=population_field.unit,
            numerator_source_system=(
                death_count_field.source[0] if death_count_field.source else None
            ),
            denominator_source_system=(
                population_field.source[0] if population_field.source else None
            ),
        ),
        registry_index,
    )

    if not legality.legal:
        raise ValueError(
            "Illegal crude mortality-rate transformation: "
            f"failed_terms={legality.failed_terms}; warnings={legality.warnings}"
        )

    required_num = {"year", "municipality_cod6", "death_count"}
    required_den = {"year", "municipality_cod6", "population"}

    missing_num = sorted(required_num - set(death_count_data.columns))
    missing_den = sorted(required_den - set(population_data.columns))

    if missing_num:
        raise ValueError(f"Death-count data missing required columns: {missing_num}")
    if missing_den:
        raise ValueError(f"Population data missing required columns: {missing_den}")

    num = death_count_data.select(
        [
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("municipality_cod6").cast(pl.Utf8, strict=False),
            pl.col("death_count").cast(pl.Float64, strict=False),
        ]
    )

    den = population_data.select(
        [
            pl.col("year").cast(pl.Int32, strict=False),
            pl.col("municipality_cod6").cast(pl.Utf8, strict=False),
            pl.col("population").cast(pl.Float64, strict=False),
        ]
    )

    joined = num.join(
        den,
        on=["year", "municipality_cod6"],
        how="left",
    )

    missing_denom = joined.filter(
        pl.col("population").is_null() | (pl.col("population") <= 0)
    )

    valid = joined.filter(
        pl.col("population").is_not_null() & (pl.col("population") > 0)
    ).with_columns(
        (pl.col("death_count") / pl.col("population")).alias("mortality_rate"),
        (pl.col("death_count") / pl.col("population") * scale).alias("mortality_rate_scaled"),
        pl.lit(scale).alias("scale"),
        pl.lit("deaths_per_100k_person_years").alias("unit"),
    )

    valid = valid.select(
        [
            "year",
            "municipality_cod6",
            "death_count",
            "population",
            "mortality_rate",
            "mortality_rate_scaled",
            "scale",
            "unit",
        ]
    ).sort(["year", "municipality_cod6"])

    warnings: list[WarningRecord] = []

    denom_missingness = missing_denom.height / joined.height if joined.height else 1.0

    if missing_denom.height:
        warnings.append(
            WarningRecord(
                field_id=None,
                source="compile_crude_mortality_rate_field",
                severity="warning" if denom_missingness < 0.20 else "severe",
                code="mortality_rate_rows_missing_denominator",
                message=(
                    f"{missing_denom.height}/{joined.height} death-count rows lacked "
                    "a valid population denominator and were excluded."
                ),
            )
        )

    parent_states = {death_count_field.state, population_field.state}
    state = "verified"

    if "illegal_excluded" in parent_states:
        state = "illegal_excluded"
    elif "fragile" in parent_states or denom_missingness >= 0.05:
        state = "fragile"

    dashboard_safe: bool | str = True if state == "verified" else "warning"

    lineage = Lineage(
        parent_ids=[death_count_field.id, population_field.id],
        operator_type="RN",
        operator_params={
            "role": "mortality_rate",
            "scale": scale,
            "formula": "death_count / population * scale",
            "join_axes": ["year", "municipality_cod6"],
        },
        registry_versions=registry_hashes,
    )

    field = FieldNode.from_lineage(
        name="Crude mortality rate by municipality and year",
        kind="intensive_density",
        carrier="Deaths/Population",
        unit="deaths_per_100k_person_years",
        support={
            "time": "year",
            "geography": "municipality",
            "spatial_code": "municipality_cod6",
        },
        axes={
            "T": "year",
            "S*": "municipality",
        },
        aggregation="rate_requires_numerator_denominator",
        role=["outcome", "mortality_rate"],
        source=sorted(set(death_count_field.source + population_field.source)),
        operator="RN",
        provenance=sorted(set(death_count_field.provenance + population_field.provenance)),
        state=state,
        dashboard_safe=dashboard_safe,
        warnings=[*legality.warnings, *[warning.code for warning in warnings]],
        lineage=lineage,
        materialization_state="materialized",
    )

    warnings = [warning.model_copy(update={"field_id": field.id}) for warning in warnings]

    n_events = float(valid["death_count"].sum()) if valid.height else 0.0
    n_denom = float(valid["population"].sum()) if valid.height else None

    q_state = QState(
        field_id=field.id,
        n_events=n_events,
        n_denom=n_denom,
        n_eff=float(valid.height),
        cov_S=None,
        cov_T=None,
        missingness=denom_missingness,
        zero_inflation=None,
        denom_fragility=denom_missingness,
        cv=None,
        moran_i=None,
        temporal_roughness=None,
        spatial_entropy=None,
        provenance_risk=max(death_count_field.provenance.count("latent"), 0) * 0.8
        if "latent" in field.provenance
        else 0.20,
        race_bridge_cv=None,
        sensitivity_width=None,
        state=state,
        dashboard_safe=dashboard_safe,
        warnings=[*legality.warnings, *[warning.code for warning in warnings]],
    )

    return CompiledField(
        field=field,
        data=valid,
        q_state=q_state,
        warnings=warnings,
    )