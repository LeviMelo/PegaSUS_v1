from __future__ import annotations

from typing import Literal

import polars as pl

from pegasus.problem1.compiled import CompiledField
from pegasus.problem1.contracts import FieldNode, Lineage, QState, WarningRecord


def compile_sim_death_count_field(
    normalized_sim: pl.DataFrame,
    *,
    registry_hashes: dict[str, str],
    geography: Literal["residence", "occurrence"] = "residence",
) -> CompiledField:
    """Compile normalized SIM-DO events into an extensive death-count field.

    This is the first real Problem 1 field materializer. It does not compute
    mortality rates. It only compiles the legal numerator field:

        SIM-DO normalized death events -> Deaths count over municipality-year.

    Rate construction will later require a compatible Population denominator
    and Delta/RN legality evaluation.
    """
    if geography == "residence":
        geo_col = "mun_residence_cod6"
        geo_state_col = "mun_residence_state"
        geo_basis = "municipality_residence"
    else:
        geo_col = "mun_occurrence_cod6"
        geo_state_col = "mun_occurrence_state"
        geo_basis = "municipality_occurrence"

    required = {"death_date", "death_date_state", geo_col, geo_state_col}
    missing = sorted(required - set(normalized_sim.columns))
    if missing:
        raise ValueError(f"Normalized SIM table missing required columns: {missing}")

    total_rows = normalized_sim.height

    valid_events = normalized_sim.filter(
        (pl.col("death_date_state") == "valid")
        & (pl.col(geo_state_col) == "valid")
        & pl.col("death_date").is_not_null()
        & pl.col(geo_col).is_not_null()
    ).with_columns(
        pl.col("death_date").str.slice(0, 4).cast(pl.Int32).alias("year"),
        pl.col(geo_col).alias("municipality_cod6"),
    )

    excluded_rows = total_rows - valid_events.height
    missingness = excluded_rows / total_rows if total_rows else 1.0

    data = (
        valid_events.group_by(["year", "municipality_cod6"])
        .agg(pl.len().alias("death_count"))
        .with_columns(
            pl.lit("SIM-DO").alias("source_system"),
            pl.lit(geo_basis).alias("geography_basis"),
        )
        .select(
            [
                "source_system",
                "geography_basis",
                "year",
                "municipality_cod6",
                "death_count",
            ]
        )
        .sort(["year", "municipality_cod6"])
    )

    warnings: list[WarningRecord] = []

    if missingness > 0:
        warnings.append(
            WarningRecord(
                field_id=None,
                source="compile_sim_death_count_field",
                severity="warning" if missingness < 0.20 else "severe",
                code="sim_events_excluded_from_count",
                message=(
                    f"{excluded_rows}/{total_rows} SIM-DO rows excluded because "
                    "death date or municipality support was invalid/missing."
                ),
            )
        )

    state = "verified" if valid_events.height > 0 and missingness < 0.10 else "fragile"
    dashboard_safe: bool | str = True if state == "verified" else "warning"

    lineage = Lineage(
        parent_ids=[],
        operator_type="event_count",
        operator_params={
            "source_system": "SIM-DO",
            "event": "death",
            "geography": geography,
            "geo_basis": geo_basis,
            "time_axis": "year",
            "measure": "death_count",
        },
        registry_versions=registry_hashes,
    )

    field = FieldNode.from_lineage(
        name=f"SIM-DO death count by {geo_basis} and year",
        kind="extensive_measure",
        carrier="Deaths",
        unit="counts",
        support={
            "time": "year",
            "geography": geo_basis,
            "spatial_code": "municipality_cod6",
        },
        axes={
            "T": "year",
            "S*": geo_basis,
        },
        aggregation="additive",
        role=["outcome", "death_numerator"],
        source=["SIM-DO"],
        operator="event_count",
        provenance=["official", "harmonized"],
        state=state,
        dashboard_safe=dashboard_safe,
        warnings=[warning.code for warning in warnings],
        lineage=lineage,
        materialization_state="materialized",
    )

    warnings = [
        warning.model_copy(update={"field_id": field.id})
        for warning in warnings
    ]

    q_state = QState(
        field_id=field.id,
        n_events=float(valid_events.height),
        n_denom=None,
        n_eff=float(valid_events.height),
        cov_S=None,
        cov_T=None,
        missingness=missingness,
        zero_inflation=None,
        denom_fragility=None,
        cv=None,
        moran_i=None,
        temporal_roughness=None,
        spatial_entropy=None,
        provenance_risk=0.20,
        race_bridge_cv=None,
        sensitivity_width=None,
        state=state,
        dashboard_safe=dashboard_safe,
        warnings=[warning.code for warning in warnings],
    )

    return CompiledField(
        field=field,
        data=data,
        q_state=q_state,
        warnings=warnings,
    )