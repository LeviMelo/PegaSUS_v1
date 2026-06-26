"""SIDRA contextual field admission into SHE (MSD §2.9, §3.10.7 V_X).

Turns acquired SIDRA *context* facts (the curated socioeconomic compendium — GDP, income,
sanitation, education, labor) into EFG ``context_gradient`` fields. Each variable is routed
through the §2.9 regime classifier, which decides whether it is used directly, deflated,
harmonized, ST-DFM bounded-interpolated, or left cross-sectional. Latent-derived regimes
carry the §3.6 provenance quarantine (dashboard-unsafe by default).

This is the read/admission boundary only. ST-DFM reconstruction itself (§2.10) is invoked
downstream when a variable's regime is ``bounded_interpolate`` and the gate holds (needs
≥3 temporal points). Single-period context (the current data scope) classifies as
``direct``/``deflate``/``cross_sectional`` and is admitted as an observed context gradient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.schemas import FieldNode
from pegasus.efg.lineage import lineage_hash, make_lineage
from pegasus.efg.node import make_field_node
from pegasus.sidra.regime import classify_sidra_context_regime
from pegasus.she.sidra_projection import project_and_bound_context_facts
from pegasus.she.stdfm.pipeline import run_stdfm_pipeline
from pegasus.she.stdfm.schema import STDFMProblem, build_stdfm_input_schema


# Population anchor (handled separately as the denominator, §2.8) — never a context field.
ANCHOR_TABLE_ID = "9606"
ANCHOR_VARIABLE_ID = "93"

# Regimes whose values are statistically reconstructed/transformed and therefore carry the
# §3.6 latent-provenance quarantine (dashboard-unsafe by default).
_LATENT_REGIMES = {"bounded_interpolate"}


def _dynamics_for_unit(unit: str) -> str:
    u = (unit or "").strip().lower()
    if u in {"%", "percent"}:
        return "proportion"
    if u in {"r$", "milr$", "sm"}:
        return "positive"
    return "continuous"


def _classify(unit: str, temporal_points: int, *, missing_t: bool) -> Any:
    # The ST-DFM gate is meaningful only when the concept has longitudinal support and
    # missing cells to reconstruct. Complete stable panels remain direct/deflate.
    return classify_sidra_context_regime(
        missing_t=missing_t,
        schema_stable=not missing_t,
        schema_mismatch=False,
        projectable=True,
        unit=unit,
        anchors_bounded=True,
        concept_compatible=True,
        temporal_points=temporal_points,
        dynamics=_dynamics_for_unit(unit),
    )


def _stdfm_transform(unit: str) -> str:
    dynamics = _dynamics_for_unit(unit)
    if dynamics == "proportion":
        return "proportion_logit"
    if dynamics == "positive":
        return "log"
    return "identity"


def _run_stdfm_for_context(
    sub: pl.DataFrame,
    *,
    field_stub: str,
    unit: str,
    projection_metadata: dict[str, Any],
    output_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    localities = sorted(str(x) for x in sub.get_column("locality_id").drop_nulls().unique().to_list())
    periods = sorted(str(x) for x in sub.get_column("period").drop_nulls().unique().to_list())
    locality_index = {value: i for i, value in enumerate(localities)}
    period_index = {value: i for i, value in enumerate(periods)}
    observations = [0.0] * (len(localities) * len(periods))
    observed = [False] * (len(localities) * len(periods))
    numeric = sub.filter((pl.col("value_status").cast(pl.Utf8) == "numeric") & pl.col("value_numeric").is_not_null())
    for row in numeric.iter_rows(named=True):
        s = locality_index.get(str(row["locality_id"]))
        t = period_index.get(str(row["period"]))
        if s is None or t is None:
            continue
        idx = s * len(periods) + t
        observations[idx] = float(row["value_numeric"])
        observed[idx] = True
    input_schema = build_stdfm_input_schema(
        field_id=field_stub,
        concept_id=field_stub,
        support={"localities": localities, "periods": periods},
        periods=periods,
        localities=localities,
        transform=_stdfm_transform(unit),
        dynamics=_dynamics_for_unit(unit),
        projection_matrix_id=str(projection_metadata.get("projection_matrix_id") or ""),
        stitch_metadata={"status": "single_segment", "segments": []},
        warnings=warnings,
    )
    problem = STDFMProblem(
        field_ids=(field_stub,),
        shape=(len(localities), len(periods), 1),
        observations=tuple(observations),
        observed_mask=tuple(observed),
        link_function_by_field=(input_schema.transform,),
        gamma_spatial=0.0,
        multi_starts=3,
    )
    result = run_stdfm_pipeline(
        input_schema,
        problem,
        output_dir=output_dir,
        require_cuda=False,
        prefer_cuda=False,
        max_iterations=500,
    )
    return {
        "output": result.output.as_manifest(),
        "certification": result.certification.as_manifest(),
        "promotable_latent_context": result.certification.status in {"verified", "fragile"},
    }


def build_sidra_context_fields(
    facts_path: str | Path,
    *,
    artifact_hash: str | None = None,
    source_manifest_hash: str | None = None,
) -> list[FieldNode]:
    """Build one ``context_gradient`` FieldNode per (table_id, variable_id) context variable."""
    original_facts_path = Path(facts_path)
    projection = project_and_bound_context_facts(original_facts_path)
    facts_path = Path(projection.path)
    frame = pl.read_parquet(facts_path)
    if frame.height == 0:
        return []

    fields: list[FieldNode] = []
    group_cols = [c for c in ("table_id", "variable_id") if c in frame.columns]
    if not group_cols:
        return []

    for keys, sub in frame.group_by(group_cols, maintain_order=True):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        table_id = str(key_map.get("table_id", ""))
        variable_id = str(key_map.get("variable_id", ""))
        if table_id == ANCHOR_TABLE_ID and variable_id == ANCHOR_VARIABLE_ID:
            continue  # population anchor is the denominator, not a context gradient

        units = [u for u in sub.get_column("unit").unique().to_list() if u] if "unit" in sub.columns else []
        unit = str(units[0]) if units else "raw_sidra_value"
        periods = sub.get_column("period").unique().to_list() if "period" in sub.columns else []
        temporal_points = len([p for p in periods if p is not None])
        n_localities = int(sub.get_column("locality_id").n_unique()) if "locality_id" in sub.columns else 0
        numeric_support = (
            sub.filter((pl.col("value_status").cast(pl.Utf8) == "numeric") & pl.col("value_numeric").is_not_null()).height
            if {"value_status", "value_numeric"} <= set(sub.columns)
            else sub.height
        )
        expected_support = max(1, n_localities) * max(1, temporal_points)
        missing_t = temporal_points >= 3 and numeric_support < expected_support

        regime = _classify(unit, temporal_points, missing_t=missing_t)
        warnings = list(dict.fromkeys([*projection.metadata.get("warnings", []), *regime.warnings]))
        stdfm_metadata: dict[str, Any] | None = None
        if regime.regime in _LATENT_REGIMES and regime.stdfm_gate:
            field_stub = f"sidra_context_{table_id}_{variable_id}"
            stdfm_metadata = _run_stdfm_for_context(
                sub,
                field_stub=field_stub,
                unit=unit,
                projection_metadata=projection.metadata,
                output_dir=facts_path.parent / "stdfm" / field_stub,
                warnings=warnings,
            )
            warnings.extend((stdfm_metadata.get("certification") or {}).get("warnings") or [])
        is_latent = bool(stdfm_metadata and stdfm_metadata.get("promotable_latent_context"))

        support: dict[str, Any] = {
            "support_kind": "sidra_context_facts",
            "source_system": "SIDRA",
            "artifact_path": str(original_facts_path),
            "sidra_facts_path": str(facts_path),
            "sidra_raw_facts_path": str(original_facts_path),
            "table_id": table_id,
            "variable_id": variable_id,
            "unit_raw": unit,
            "n_localities": n_localities,
            "temporal_points": temporal_points,
            "numeric_support": int(numeric_support),
            "expected_support": int(expected_support),
            "sidra_regime": regime.regime,
            "stdfm_gate": bool(regime.stdfm_gate),
            "regime_reason": regime.reason,
            "projection": projection.metadata,
        }
        if stdfm_metadata is not None:
            support["stdfm"] = stdfm_metadata
        axes = {
            "geography_axis": "N6",
            "time_axis": "period",
            "context_variable": f"{table_id}:{variable_id}",
            "sidra_regime": regime.regime,
            "projection_matrix_id": projection.metadata.get("projection_matrix_id"),
        }
        if is_latent:
            warnings.append("latent")  # §3.6 latent provenance quarantine
        elif stdfm_metadata is not None:
            warnings.append("stdfm_executed_not_certified_for_latent_promotion")
        provenance = ["SHE_SubstrateBundle", "sidra_context", f"sidra_regime_{regime.regime}"]
        if is_latent:
            provenance.append("latent")
        lineage = make_lineage(
            parent_ids=[],
            operator_type="sidra_context_field",
            operator_params=dict(support),
            registry_versions={
                "SIDRA": str(artifact_hash or source_manifest_hash or "unknown"),
                "sidra_context_regime": "regime_v1",
                "sidra_projection": str(projection.metadata.get("projection_matrix_id") or "unknown"),
            },
            source_manifest_hashes=[h for h in (source_manifest_hash, artifact_hash) if h],
            code_version="sidra_context_v1",
        )
        field = make_field_node(
            name=f"sidra_context_{table_id}_{variable_id}",
            kind="latent_context" if is_latent else "context_gradient",
            carrier="ContextCells",
            unit=unit,
            support=support,
            axes=axes,
            aggregation="non_aggregable",
            role=["sidra_context", "context_gradient_seed", "covariate", "source_field"],
            source=["SIDRA", str(facts_path), f"SIDRA_{table_id}_{variable_id}"],
            operator="sidra_context_field",
            provenance=provenance,
            state="warning" if is_latent else "fragile",
            warnings=warnings,
            lineage=lineage,
            materialization_state="metadata_only",
            path=None,
            dashboard_safe="warning",
        ).model_copy(update={"id": f"sidra_context_{table_id}_{variable_id}_{lineage_hash(lineage)[:16]}"})
        fields.append(field)
    return fields


__all__ = ["build_sidra_context_fields", "ANCHOR_TABLE_ID", "ANCHOR_VARIABLE_ID"]
