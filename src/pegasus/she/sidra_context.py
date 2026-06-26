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


def _classify(unit: str, temporal_points: int) -> Any:
    # Single-table, single-acquisition context: schema is stable, no break, projectable.
    # temporal_points drives the ST-DFM gate (needs >= 3); fewer => direct/deflate/cross.
    return classify_sidra_context_regime(
        missing_t=temporal_points < 1,
        schema_stable=True,
        schema_mismatch=False,
        projectable=True,
        unit=unit,
        anchors_bounded=True,
        concept_compatible=True,
        temporal_points=temporal_points,
        dynamics=_dynamics_for_unit(unit),
    )


def build_sidra_context_fields(
    facts_path: str | Path,
    *,
    artifact_hash: str | None = None,
    source_manifest_hash: str | None = None,
) -> list[FieldNode]:
    """Build one ``context_gradient`` FieldNode per (table_id, variable_id) context variable."""
    facts_path = Path(facts_path)
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

        regime = _classify(unit, temporal_points)
        is_latent = regime.regime in _LATENT_REGIMES and regime.stdfm_gate

        support: dict[str, Any] = {
            "support_kind": "sidra_context_facts",
            "source_system": "SIDRA",
            "artifact_path": str(facts_path),
            "sidra_facts_path": str(facts_path),
            "table_id": table_id,
            "variable_id": variable_id,
            "unit_raw": unit,
            "n_localities": int(sub.get_column("locality_id").n_unique()) if "locality_id" in sub.columns else 0,
            "temporal_points": temporal_points,
            "sidra_regime": regime.regime,
            "stdfm_gate": bool(regime.stdfm_gate),
            "regime_reason": regime.reason,
        }
        axes = {
            "geography_axis": "N6",
            "time_axis": "period",
            "context_variable": f"{table_id}:{variable_id}",
            "sidra_regime": regime.regime,
        }
        warnings = list(regime.warnings)
        if is_latent:
            warnings.append("latent")  # §3.6 latent provenance quarantine
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
            },
            source_manifest_hashes=[h for h in (source_manifest_hash, artifact_hash) if h],
            code_version="sidra_context_v1",
        )
        field = make_field_node(
            name=f"sidra_context_{table_id}_{variable_id}",
            kind="context_gradient",
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
