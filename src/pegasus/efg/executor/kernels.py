"""Physical tensor executor kernels for autonomous EFG.

Operator kernels that materialize tensors (count/functional/sum/RN/bridge/SIDRA/
population-solver). Pure code motion from the former monolithic executor module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from pegasus.core.enums import MaterializationState
from pegasus.core.schemas import FieldNode
from pegasus.efg.dag import EFGResult
from pegasus.efg.race_bridge import (
    bridge_admin_race_group_counts,
    load_race_bridge_prior,
)

from pegasus.efg.executor.support import *


def _count_tensor(field: FieldNode, source: Path) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source), field)
    base_keys = _support_keys(df)
    support = _as_dict(field.support)
    conditions = support.get("restrict_conditions")
    if conditions:
        df = _apply_restrict_conditions(df, list(conditions))
    keys = list(base_keys)
    stratify_icd = support.get("stratify_icd")
    if stratify_icd:
        axis_name = str(support.get("icd_axis") or ("icd_chapter" if stratify_icd == "chapter" else "icd_block"))
        icd_column = str(support.get("icd_column") or "")
        df = _add_icd_stratum(df, icd_column, str(stratify_icd), axis_name)
        keys = [*keys, axis_name]
    # General demographic stratification (MSD §3.7.4): group by a canonical axis derived
    # from a source column, mapping raw codes -> canonical categories so the count joins a
    # matching demographic population denominator. Unknown/total categories are dropped.
    stratify_column = support.get("stratify_column")
    if stratify_column:
        from pegasus.registries.demographic_axis import (
            TOTAL,
            UNKNOWN,
            age_group_for_years,
            canonical_categories,
            source_category_map,
        )

        axis_name = str(support.get("stratify_axis") or stratify_column)
        source_system = str(support.get("stratify_source") or "")
        raw_column = str(stratify_column)
        if raw_column in df.columns:
            tmp = "__canonical_stratum__"
            if axis_name == "age_group":
                # Age is a direct arithmetic bucketing of the source's single-year age
                # field (SIM/SINASC/SIH carry age_years / maternal_age_years), NOT a
                # category-code crosswalk (MSD §3.7.4). Bucket to the canonical age_N
                # basis so the count joins the single-year population denominator.
                df = df.with_columns(
                    pl.col(raw_column)
                    .map_elements(age_group_for_years, return_dtype=pl.Utf8)
                    .alias(axis_name)
                ).filter(~pl.col(axis_name).is_in([TOTAL, UNKNOWN]))
                keys = [*keys, axis_name]
            elif axis_name == "race":
                prior_path = support.get("race_bridge_prior_path")
                if prior_path:
                    # Bridge_R configured: redistribute this cell's admin race codes onto
                    # the census self-declared race axis (MSD §3.7.4/§2.8.6) per (geo,time)
                    # group, so the SELF-DECLARED race count divides the self-declared
                    # population. Posterior counts are a bridge estimate, NOT a raw
                    # observation -- the field carries the bridge's epistemic warnings.
                    from pegasus.efg.race_bridge import (
                        bridge_admin_race_group_counts,
                        load_race_bridge_prior,
                    )

                    prior = load_race_bridge_prior(str(prior_path))
                    state_col = "race_missingness_state" if "race_missingness_state" in df.columns else None
                    bridged_rows: list[dict[str, Any]] = []
                    for support_values, group in _fixedc_support_groups(df):
                        codes = group[raw_column].to_list()
                        states = group[state_col].to_list() if state_col is not None else None
                        posterior = bridge_admin_race_group_counts(
                            race_codes=codes, race_states=states, prior=prior,
                            support={**support_values, "n_events": int(group.height)},
                        )
                        for target in prior.target_categories:
                            bridged_rows.append({
                                **support_values,
                                axis_name: str(target),
                                VALUE_COLUMN: float(posterior.posterior_counts[target]),
                            })
                    schema = {**{k: df.schema[k] for k in base_keys}, axis_name: pl.Utf8, VALUE_COLUMN: pl.Float64}
                    out = pl.DataFrame(bridged_rows, schema=schema) if bridged_rows else pl.DataFrame(schema=schema)
                    return out.with_columns([
                        pl.lit(field.id).alias("field_id"),
                        pl.lit(field.name).alias("field_name"),
                        pl.lit(field.operator or "count_measure").alias("operator"),
                    ])
                # No bridge prior: administrative race/color is NOT self-declared census
                # race -- a direct code->canonical crosswalk would be silent redistribution
                # (§3.7.4). Keep the RAW admin code; align_fields gates the rate on
                # race_bridge_required so this never divides a self-declared population.
                df = df.with_columns(
                    pl.col(raw_column).cast(pl.Utf8, strict=False).alias(axis_name)
                ).filter(pl.col(axis_name).is_not_null())
                keys = [*keys, axis_name]
            else:
                # sex (and any future direct-crosswalk axis). Accept BOTH raw source codes
                # and already-canonical values: the normalizers may emit the canonical
                # category directly (SIM/SINASC/SIH 'sex' is 'male'/'female', not '1'/'2'),
                # so identity on the canonical vocabulary keeps a stale/no-op code map from
                # dropping every row into __unknown__.
                code_map = source_category_map(axis_name, source_system)
                mapping = {**{c: c for c in canonical_categories(axis_name)}, **code_map}
                if mapping:
                    map_df = pl.DataFrame(
                        {raw_column: list(mapping.keys()), tmp: list(mapping.values())},
                        schema={raw_column: pl.Utf8, tmp: pl.Utf8},
                    )
                    df = (
                        df.with_columns(pl.col(raw_column).cast(pl.Utf8, strict=False))
                        .join(map_df, on=raw_column, how="left")
                        .with_columns(pl.col(tmp).fill_null(UNKNOWN).alias(axis_name))
                        .filter(~pl.col(axis_name).is_in([TOTAL, UNKNOWN]))
                        .drop(tmp)
                    )
                    keys = [*keys, axis_name]
    if keys:
        out = df.group_by(keys).agg(pl.len().cast(pl.Float64).alias(VALUE_COLUMN)).sort(keys)
    else:
        out = pl.DataFrame({VALUE_COLUMN: [float(df.height)]})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "count_measure").alias("operator"),
    ])
    return out


def _functional_tensor(field: FieldNode, source: Path) -> pl.DataFrame:
    """Materialize a statistical functional (mean/median) of a per-record mark over each
    support cell (MSD §3.10.4-6 Ψ operators)."""
    df = _support_frame(pl.read_parquet(source), field)
    support = _as_dict(field.support)
    mark = str(support.get("mark_column") or "")
    functional = str(support.get("functional") or "mean")
    if mark not in df.columns:
        raise ValueError(f"functional field {field.id} mark column {mark!r} absent from source")
    df = df.with_columns(pl.col(mark).cast(pl.Float64, strict=False).alias("__mark__"))
    agg = pl.col("__mark__").median() if functional == "median" else pl.col("__mark__").mean()
    keys = _support_keys(df)
    if keys:
        out = df.group_by(keys).agg(agg.cast(pl.Float64).alias(VALUE_COLUMN)).sort(keys)
    else:
        scalar = df.get_column("__mark__").median() if functional == "median" else df.get_column("__mark__").mean()
        out = pl.DataFrame({VALUE_COLUMN: [float(scalar) if scalar is not None else None]})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(f"psi_{functional}").alias("operator"),
    ])
    return out


def _sum_tensor(field: FieldNode, source: Path, column: str) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source), field)
    df = df.with_columns(pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0).alias("__value__"))
    keys = _support_keys(df)
    if keys:
        out = df.group_by(keys).agg(pl.col("__value__").sum().cast(pl.Float64).alias(VALUE_COLUMN)).sort(keys)
    else:
        out = pl.DataFrame({VALUE_COLUMN: [float(df["__value__"].sum() or 0.0)]})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "sigma_C").alias("operator"),
    ])
    return out


def _source_field_tensor(field: FieldNode, source: Path, column: str) -> pl.DataFrame:
    df = _support_frame(pl.read_parquet(source), field)
    keys = _support_keys(df)
    out = df.select([
        *(pl.col(key) for key in keys),
        pl.col(column).alias(VALUE_COLUMN),
    ]).with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "source_field").alias("operator"),
    ])
    return out


def _load_parent_tensor(parent: FieldNode) -> pl.DataFrame:
    if not parent.path:
        raise ValueError(f"parent {parent.id} has no materialized path")
    path = Path(parent.path)
    if not path.exists():
        raise FileNotFoundError(f"parent tensor missing: {path}")
    df = pl.read_parquet(path)
    if VALUE_COLUMN not in df.columns:
        raise ValueError(f"parent tensor lacks {VALUE_COLUMN!r}: {path}")
    return df


_YEAR_KEYS = ("year", "admission_year", "birth_year", "competence_year")


def _temporal_lag_of(field: Any) -> int:
    """Year lag declared for a bridge field (support/operator_params), else 0."""
    support = _as_dict(getattr(field, "support", {}) or {})
    params = _as_dict(support.get("operator_params") or support.get("params") or {})
    for source in (support, params):
        value = source.get("temporal_lag")
        if value is not None:
            try:
                k = int(value)
            except (TypeError, ValueError):
                return 0
            return k if k > 0 else 0
    return 0


def _shift_year(df: pl.DataFrame, lag: int) -> pl.DataFrame:
    """Add `lag` to the first present year column so left(t) aligns to right(t+lag).

    Output cell year t then pairs the left operand's value from year t-lag (MSD §2.11
    delayed cross-source effect). Pure and column-agnostic across the SIH/SINASC/SIM
    year axis names."""
    if lag <= 0:
        return df
    for column in _YEAR_KEYS:
        if column in df.columns:
            return df.with_columns((pl.col(column).cast(pl.Int64, strict=False) + lag).alias(column))
    return df


def _join_keys(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    common = [column for column in left.columns if column in right.columns]
    return [column for column in common if column not in METADATA_COLUMNS and column != VALUE_COLUMN]


def _compute_rn_ratio(field: FieldNode, parents_by_id: dict[str, FieldNode], output_dir: Path) -> tuple[Path, int, dict[str, Any]]:
    parent_ids = list(field.lineage.parent_ids or [])
    if len(parent_ids) < 2:
        raise ValueError(f"RN field {field.id} requires numerator and denominator parents")
    numerator = parents_by_id[parent_ids[0]]
    denominator = parents_by_id[parent_ids[1]]
    n = _load_parent_tensor(numerator).rename({VALUE_COLUMN: "value_numerator"})
    d = _load_parent_tensor(denominator).rename({VALUE_COLUMN: "value_denominator"})
    keys = _join_keys(n, d)

    # Ecological Fallacy Guard: Aggregate numerator up to denominator's spatial support if mismatched.
    if "municipality_cod6" in n.columns and "municipality_cod6" not in d.columns:
        agg_keys = [k for k in keys if k != "municipality_cod6"]
        if agg_keys:
            n = n.group_by(agg_keys).agg(pl.col("value_numerator").sum())
        else:
            n = pl.DataFrame({"value_numerator": [n["value_numerator"].sum()]})
        keys = _join_keys(n, d)

    if not keys:
        raise RuntimeError(
            "RN operator requires at least one intersecting support axis; "
            "cross-join is forbidden to prevent OOM and indicates failed Δ support alignment."
        )
    # Stratifier columns present on the numerator but absent from the (unstratified)
    # denominator — e.g. icd_chapter/icd_block from a σ_C restriction. They are NOT join
    # keys (the denominator broadcasts across strata) but MUST survive into the output,
    # otherwise cause-specific rates collapse to one ambiguous row per (year, municipality).
    numerator_strata = [
        column
        for column in n.columns
        if column not in keys
        and column != "value_numerator"
        and column not in METADATA_COLUMNS
    ]
    joined = n.join(d, on=keys, how="left", suffix="_denominator")

    missing_denom_count = joined.filter(pl.col("value_denominator").is_null() | pl.col("value_denominator").is_nan()).height
    denom_fragility = float(missing_denom_count) / float(joined.height) if joined.height > 0 else 1.0

    parent_denom_id = parent_ids[1]
    parent_fragility = float(parents_by_id[parent_denom_id].support.get("denom_fragility", 0.0))
    combined_fragility = min(1.0, parent_fragility + denom_fragility)

    out = joined.with_columns(
        pl.when((pl.col("value_denominator") > 0) & pl.col("value_denominator").is_not_null())
        .then(pl.col("value_numerator") / pl.col("value_denominator"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias(VALUE_COLUMN)
    )

    keep = [column for column in [*keys, *numerator_strata] if column in out.columns]
    out = out.select([
        *[pl.col(column) for column in keep],
        pl.col(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("RN").alias("operator"),
    ])
    path, rows = _write(output_dir / f"{field.id}.parquet", out)
    # Report the realized numerator/denominator totals so the Q-tensor (MSD §3.12)
    # carries the true event/denominator counts for this ratio rather than 0/None.
    num_total = float(n.select(pl.col("value_numerator").sum()).item() or 0.0)
    den_total = float(d.select(pl.col("value_denominator").sum()).item() or 0.0)
    # EFG-OUT-01 (MSD-III §II.3): emit the count+exposure MeasuredQuantity as a sidecar
    # (the rate parquet above is the *derived* view). Preserves count-variance and the
    # log(exposure) offset so the LDO models count directly rather than reversing a rate.
    from pegasus.efg.measured_quantity import measured_quantity_from_rn_join, write_measured_quantity
    mq = measured_quantity_from_rn_join(
        joined, keys=keys, strata=numerator_strata, field_id=field.id,
        provenance=field.provenance, denom_fragility=combined_fragility, axes=dict(field.axes or {}),
    )
    mq_path = write_measured_quantity(mq, output_dir / f"{field.id}.measured_quantity.parquet")
    return path, rows, {
        "denom_fragility": combined_fragility,
        "n_events": num_total,
        "n_denom": den_total if den_total > 0 else None,
        "offset_semantics": "log_exposure",
        "measured_quantity_ref": str(mq_path),
    }



def _is_bridge_divergence(field: FieldNode) -> bool:
    op = str(field.operator or "").lower()
    kind = str(field.kind or "").lower()
    support = _as_dict(field.support)
    params = _as_dict(support.get("operator_params") or support.get("params") or {})
    haystack = " ".join([
        op,
        kind,
        str(support.get("bridge_type") or "").lower(),
        str(params.get("bridge_type") or "").lower(),
        str(params.get("operator") or "").lower(),
    ])
    return (
        "divergence" in haystack
        or "morbidity_mortality" in haystack
        or "mortality_morbidity" in haystack
        or op in {"bridge_divergence", "divergence_log_ratio"}
    )


def _is_race_bridge(field: FieldNode) -> bool:
    support = _as_dict(field.support)
    params = _as_dict(field.lineage.operator_params)
    return (
        field.operator == "Bridge_R_fixedC_dynamic_weight"
        or field.operator == "Bridge_R_localPi_posteriorC"
        or support.get("bridge_operator") == "Bridge_R_fixedC_dynamic_weight"
        or support.get("bridge_operator") == "Bridge_R_localPi_posteriorC"
        or params.get("bridge_operator") == "Bridge_R_fixedC_dynamic_weight"
        or params.get("bridge_operator") == "Bridge_R_localPi_posteriorC"
    )


def _race_column(field: FieldNode, parent: FieldNode, df: pl.DataFrame) -> str:
    support = _as_dict(parent.support)
    candidates = [
        support.get("column"),
        support.get("source_column"),
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate) in df.columns:
            return str(candidate)
    raise ValueError(
        f"Bridge_R field {field.id} parent {parent.id} lacks a registry-declared "
        "administrative race source column present in the source artifact"
    )


def _fixedc_support_groups(df: pl.DataFrame) -> list[tuple[dict[str, Any], pl.DataFrame]]:
    keys = _support_keys(df)
    if not keys:
        return [({}, df)]
    rows = df.select(keys).unique().sort(keys).iter_rows(named=True)
    groups: list[tuple[dict[str, Any], pl.DataFrame]] = []
    for values in rows:
        sub = df
        for key, value in values.items():
            if value is None:
                sub = sub.filter(pl.col(key).is_null())
            else:
                sub = sub.filter(pl.col(key) == value)
        groups.append((dict(values), sub))
    return groups


def _compute_race_bridge_tensor(
    field: FieldNode,
    parent: FieldNode,
    output_dir: Path,
) -> tuple[Path, int, dict[str, Any]]:
    params = {**_as_dict(field.support), **_as_dict(field.lineage.operator_params)}
    prior_path = params.get("prior_path") or params.get("bridge_prior_path")
    if not prior_path:
        raise ValueError(f"Bridge_R field {field.id} missing prior_path")
    prior = load_race_bridge_prior(prior_path)

    source = _source_path(parent)
    if source is None:
        raise ValueError(f"Bridge_R parent {parent.id} has no source artifact path")
    df = _support_frame(pl.read_parquet(source), parent)
    race_column = _race_column(field, parent, df)
    state_col = "race_missingness_state" if "race_missingness_state" in df.columns else None

    rows: list[dict[str, Any]] = []
    summary_missing = 0
    summary_total = 0
    summary_cv: list[float] = []
    summary_width: list[float] = []
    for support_values, group in _fixedc_support_groups(df):
        race_values = group[race_column].to_list()
        states = group[state_col].to_list() if state_col is not None else None
        posterior = bridge_admin_race_group_counts(
            race_codes=race_values,
            race_states=states,
            prior=prior,
            support={**support_values, "n_events": int(group.height)},
        )
        raw_counts = posterior.raw_admin_counts
        metadata = posterior.metadata()
        summary_missing += posterior.missing_count
        summary_total += int(group.height)
        summary_cv.append(float(posterior.race_bridge_cv))
        summary_width.append(float(posterior.sensitivity_width))
        for target in prior.target_categories:
            rows.append({
                **support_values,
                "target_race_category": target,
                VALUE_COLUMN: float(posterior.posterior_counts[target]),
                "lower_count": float(posterior.lower_counts[target]),
                "upper_count": float(posterior.upper_counts[target]),
                "raw_admin_counts_json": json.dumps(raw_counts, sort_keys=True),
                "missing_count": int(posterior.missing_count),
                "missing_race_share": float(posterior.missing_share),
                "race_bridge_cv": float(posterior.race_bridge_cv),
                "sensitivity_width": float(posterior.sensitivity_width),
                "prior_hash": prior.prior_hash,
                "bridge_mode": prior.mode,
                "bridge_operator": "Bridge_R_localPi_posteriorC",
                "field_id": field.id,
                "field_name": field.name,
                "operator": "Bridge_R_localPi_posteriorC",
                "bridge_metadata_json": json.dumps(metadata, sort_keys=True, default=str),
            })

    out = pl.DataFrame(rows) if rows else pl.DataFrame({
        "field_id": [field.id],
        "field_name": [field.name],
        "operator": ["Bridge_R_localPi_posteriorC"],
        VALUE_COLUMN: [0.0],
        "lower_count": [0.0],
        "upper_count": [0.0],
        "missing_race_share": [0.0],
        "race_bridge_cv": [0.0],
        "sensitivity_width": [prior.sensitivity_width],
        "prior_hash": [prior.prior_hash],
        "bridge_mode": [prior.mode],
        "bridge_operator": ["Bridge_R_localPi_posteriorC"],
    })
    path, row_count = _write(output_dir / f"{field.id}.parquet", out)
    metadata = {
        "missing_race_share": (summary_missing / float(summary_total)) if summary_total else 0.0,
        "race_bridge_cv": max(summary_cv) if summary_cv else 0.0,
        "sensitivity_width": max(summary_width) if summary_width else prior.sensitivity_width,
        "prior_hash": prior.prior_hash,
        "bridge_mode": prior.mode,
        "bridge_operator": "Bridge_R_localPi_posteriorC",
        "raw_admin_counts_preserved": True,
        "missing_category_preserved": True,
    }
    return path, row_count, metadata


def _compute_bridge_tensor(field: FieldNode, parents_by_id: dict[str, FieldNode], output_dir: Path) -> tuple[Path, int, dict[str, Any] | None]:
    parent_ids = list(field.lineage.parent_ids or [])
    if not parent_ids:
        out = pl.DataFrame({
            "field_id": [field.id],
            "field_name": [field.name],
            "operator": [field.operator or "bridge"],
            VALUE_COLUMN: [None],
        }).cast({VALUE_COLUMN: pl.Float64})
        path, rows = _write(output_dir / f"{field.id}.parquet", out)
        return path, rows, None

    if _is_race_bridge(field):
        return _compute_race_bridge_tensor(field, parents_by_id[parent_ids[0]], output_dir)

    if _is_bridge_divergence(field) and len(parent_ids) == 2:
        p0 = _load_parent_tensor(parents_by_id[parent_ids[0]])
        p1 = _load_parent_tensor(parents_by_id[parent_ids[1]])
        # Temporal-lag divergence (MSD §2.11): shift the left operand's year by k so
        # the output cell at year t pairs left(t-k) with right(t) — log(left(t-k)/right(t)).
        # Declared by the bridge grammar; 0 = the standard contemporaneous divergence.
        lag = _temporal_lag_of(field)
        if lag:
            p0 = _shift_year(p0, lag)
        keys = _join_keys(p0, p1)
        if not keys:
            raise RuntimeError("Bridge divergence requires intersecting support axes.")

        joined = p0.join(p1, on=keys, how="inner", suffix="_right")
        epsilon = 1e-9
        out = joined.with_columns(
            ((pl.col(VALUE_COLUMN) + epsilon) / (pl.col(VALUE_COLUMN + "_right") + epsilon))
            .log()
            .cast(pl.Float64)
            .alias(VALUE_COLUMN)
        )
        keep = [c for c in keys if c in out.columns]
        out = out.select([
            *[pl.col(c) for c in keep],
            pl.col(VALUE_COLUMN),
            pl.lit(field.id).alias("field_id"),
            pl.lit(field.name).alias("field_name"),
            pl.lit(field.operator or "divergence_log_ratio").alias("operator"),
        ])
        path, rows = _write(output_dir / f"{field.id}.parquet", out)
        return path, rows, None

    # Fallback for Bridge_R / unary bridges
    if str(field.operator or "").startswith("Bridge_R"):
        raise ValueError(f"Bridge_R operator {field.operator!r} has no physical executor")
    parent = parents_by_id[parent_ids[0]]
    df = _load_parent_tensor(parent)
    out = df.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit(field.operator or "Bridge_R").alias("operator"),
    ])
    path, rows = _write(output_dir / f"{field.id}.parquet", out)
    return path, rows, None


def _sidra_population_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize the SIDRA population denominator as a per-municipality panel
    (municipality_cod6, year, value) so the Radon-Nikodym rate join matches each
    municipality's deaths/births to its own population — the national grid."""
    support = _as_dict(field.support)
    facts_path = support.get("sidra_facts_path") or support.get("artifact_path")
    if not facts_path:
        raise ValueError("SIDRA population anchor field has no sidra_facts_path")
    from pegasus.denominators.population.anchor import load_sidra_population_totals_frame

    frame = load_sidra_population_totals_frame(facts_path)
    if frame.height == 0:
        raise ValueError("SIDRA population facts contain no total-category population rows")
    out = frame.with_columns(
        pl.col("value").cast(pl.Float64).alias(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("sidra_population_total_anchor").alias("operator"),
    )
    return _write(output_dir / f"{field.id}.parquet", out)


def _sidra_context_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize a SIDRA context gradient (§3.10.7 V_X) as a per-municipality panel
    (year, municipality_cod6, value). locality_id is cod7; municipality_cod6 = cod7[:6],
    matching the DATASUS event geography so PIRS can use it as a covariate."""
    support = _as_dict(field.support)
    facts_path = support.get("sidra_facts_path") or support.get("artifact_path")
    if not facts_path:
        raise ValueError("SIDRA context field has no facts path")
    table_id = str(support.get("table_id") or "")
    variable_id = str(support.get("variable_id") or "")
    df = pl.read_parquet(facts_path)
    if "table_id" in df.columns and table_id:
        df = df.filter(pl.col("table_id").cast(pl.Utf8) == table_id)
    if "variable_id" in df.columns and variable_id:
        df = df.filter(pl.col("variable_id").cast(pl.Utf8) == variable_id)
    value_col = "value_numeric" if "value_numeric" in df.columns else VALUE_COLUMN
    out = df.with_columns([
        pl.col("locality_id").cast(pl.Utf8).str.slice(0, 6).alias("municipality_cod6"),
        pl.col("period").cast(pl.Utf8).str.slice(0, 4).cast(pl.Int64, strict=False).alias("year"),
    ]).select([
        pl.col("year"),
        pl.col("municipality_cod6"),
        pl.col(value_col).cast(pl.Float64, strict=False).alias(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("sidra_context_field").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


def _category_code_for_classification(category_tuple_raw: Any, classification_id: str) -> str | None:
    import json
    try:
        pairs = json.loads(category_tuple_raw) if isinstance(category_tuple_raw, str) else category_tuple_raw
    except Exception:
        return None
    for pair in pairs or []:
        if pair and str(pair[0]) == str(classification_id):
            return str(pair[1])
    return None


def _sidra_demographic_population_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize a demographic-stratified population panel (MSD §2.8):
    (year, municipality_cod6, <axis>, value), axis category canonicalized via the
    demographic-axis registry, dropping the marginal Total and unknown categories."""
    from pegasus.registries.demographic_axis import TOTAL, UNKNOWN, map_category

    support = _as_dict(field.support)
    facts_path = support.get("sidra_facts_path") or support.get("artifact_path")
    classification_id = str(support.get("classification_id") or "")
    axis = str(support.get("demographic_axis") or "stratum")
    if not facts_path:
        raise ValueError("demographic population field has no facts path")
    df = pl.read_parquet(facts_path)
    value_col = "value_numeric" if "value_numeric" in df.columns else VALUE_COLUMN
    rows: list[dict[str, Any]] = []
    for record in df.iter_rows(named=True):
        code = _category_code_for_classification(record.get("category_tuple"), classification_id)
        canonical = map_category(axis, "SIDRA", code) if code is not None else UNKNOWN
        if canonical in {TOTAL, UNKNOWN}:
            continue  # marginal/unknown is not a stratum of the disaggregated tensor
        locality = str(record.get("locality_id") or "")
        period = str(record.get("period") or "")
        value = record.get(value_col)
        rows.append({
            "year": int(period[:4]) if period[:4].isdigit() else None,
            "municipality_cod6": locality[:6],
            axis: canonical,
            VALUE_COLUMN: float(value) if value is not None else None,
        })
    out = pl.DataFrame(rows) if rows else pl.DataFrame({VALUE_COLUMN: []}, schema={VALUE_COLUMN: pl.Float64})
    out = out.with_columns([
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("sidra_demographic_population").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


def _population_solver_tensor(field: FieldNode, output_dir: Path) -> tuple[Path, int]:
    """Materialize a solver-produced population tensor as an EFG denominator panel."""
    support = _as_dict(field.support)
    tensor_path = support.get("population_tensor_path") or support.get("artifact_path")
    if not tensor_path:
        raise ValueError("population solver field has no population_tensor_path")
    df = pl.read_parquet(tensor_path)
    required = {"year", "municipality_cod6", VALUE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"population solver tensor missing columns: {sorted(missing)}")
    # This field is the MARGINAL population over a single demographic axis (or the
    # crude total when None): sum the full (age,sex,race) tensor over every axis
    # except this field's, so a sex-stratified numerator divides by the sex-marginal
    # population (MSD §3.7.4). `marginal_demographic_axis` is set by the materializer.
    marginal_axis = support.get("marginal_demographic_axis")
    keep_axes = [marginal_axis] if (marginal_axis and marginal_axis in df.columns) else []
    group_keys = ["year", "municipality_cod6", *keep_axes]
    agg = (
        df.with_columns(
            pl.col("year").cast(pl.Int64, strict=False),
            pl.col("municipality_cod6").cast(pl.Utf8),
            *[pl.col(axis).cast(pl.Utf8) for axis in keep_axes],
            pl.col(VALUE_COLUMN).cast(pl.Float64, strict=False),
        )
        .group_by(group_keys)
        .agg(pl.col(VALUE_COLUMN).sum().alias(VALUE_COLUMN))
    )
    out = agg.select([
        *group_keys,
        pl.col(VALUE_COLUMN),
        pl.lit(field.id).alias("field_id"),
        pl.lit(field.name).alias("field_name"),
        pl.lit("population_tensor_solver").alias("operator"),
    ])
    return _write(output_dir / f"{field.id}.parquet", out)


__all__ = [
    "_count_tensor",
    "_functional_tensor",
    "_sum_tensor",
    "_source_field_tensor",
    "_load_parent_tensor",
    "_YEAR_KEYS",
    "_temporal_lag_of",
    "_shift_year",
    "_join_keys",
    "_compute_rn_ratio",
    "_is_bridge_divergence",
    "_is_race_bridge",
    "_race_column",
    "_fixedc_support_groups",
    "_compute_race_bridge_tensor",
    "_compute_bridge_tensor",
    "_sidra_population_tensor",
    "_sidra_context_tensor",
    "_category_code_for_classification",
    "_sidra_demographic_population_tensor",
    "_population_solver_tensor",
]
