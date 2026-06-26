"""Demographic population tensor admission (MSD §2.8).

Admits disaggregated SIDRA population facts (e.g. table 9606 by Sexo/Cor/Idade) as a
Population field carrying a canonical demographic stratifier axis (sex/race/age_group).
This is the denominator side of stratified rates: a sex-stratified death count is divided
by the sex-stratified population on the shared canonical axis (§3.7.4 alignment).

For a fully-enumerated census period the tensor is the *directly observed* disaggregated
counts (population_mode = official_sidra_anchor) — no solver. The orphaned
`she.population` block-coordinate/ADMM solvers (§2.8.3+) are for the
independent/sim-informed reconstruction modes (incomplete data), invoked elsewhere; we do
not run a solver to "reconstruct" already-observed cells.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pegasus.core.schemas import FieldNode
from pegasus.efg.lineage import lineage_hash, make_lineage
from pegasus.efg.node import make_field_node
from pegasus.registries.demographic_axis import axis_for_classification


def _detect_classification_id(frame: pl.DataFrame) -> str | None:
    """Find the (single) SIDRA classification id present in the facts' classification_tuple."""
    if "classification_tuple" not in frame.columns:
        return None
    import json

    for raw in frame.get_column("classification_tuple").drop_nulls().unique().to_list():
        try:
            pairs = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        for pair in pairs or []:
            if pair and str(pair[0]):
                return str(pair[0])
    return None


def build_sidra_demographic_population_fields(
    facts_path: str | Path,
    *,
    artifact_hash: str | None = None,
    source_manifest_hash: str | None = None,
    registry_root: str | Path = "config/registries",
) -> list[FieldNode]:
    """Build a demographic-stratified Population field from disaggregated SIDRA facts."""
    facts_path = Path(facts_path)
    frame = pl.read_parquet(facts_path)
    if frame.height == 0:
        return []
    classification_id = _detect_classification_id(frame)
    if classification_id is None:
        return []
    axis = axis_for_classification(classification_id, registry_root=registry_root)
    if axis is None:
        return []  # classification not mapped to a canonical demographic axis

    support = {
        "support_kind": "sidra_demographic_population_tensor",
        "source_system": "SIDRA",
        "artifact_path": str(facts_path),
        "sidra_facts_path": str(facts_path),
        "table_id": str(frame.get_column("table_id")[0]) if "table_id" in frame.columns else None,
        "classification_id": classification_id,
        "demographic_axis": axis,
        "PopulationTensorMode": "official_sidra_anchor",
        "population_tensor_diagnostics": f"official_sidra_disaggregated_{axis}",
    }
    axes = {
        "geography_axis": "N6",
        "time_axis": "period",
        "population_strata_axis": axis,
        axis: "stratified",
        "population_tensor_mode": "official_sidra_anchor",
    }
    lineage = make_lineage(
        parent_ids=[],
        operator_type="sidra_demographic_population",
        operator_params=dict(support),
        registry_versions={
            "SIDRA": str(artifact_hash or source_manifest_hash or "unknown"),
            "demographic_axis_maps": "v1.0",
        },
        source_manifest_hashes=[h for h in (source_manifest_hash, artifact_hash) if h],
        code_version="demographic_tensor_v1",
    )
    field = make_field_node(
        name=f"population_tensor_demographic_{axis}",
        kind="extensive_measure",
        carrier="Population",
        unit="persons",
        support=support,
        axes=axes,
        aggregation="additive",
        role=["population_tensor", "population_denominator_seed", "demographic_stratified", "source_field"],
        source=["SIDRA", str(facts_path), f"SIDRA_demographic_{axis}"],
        operator="sidra_demographic_population",
        provenance=["SHE_SubstrateBundle", "population_tensor", "official", f"demographic_{axis}"],
        state="verified",
        warnings=[],
        lineage=lineage,
        materialization_state="metadata_only",
        path=None,
        dashboard_safe=False,
    ).model_copy(update={"id": f"population_tensor_demographic_{axis}_{lineage_hash(lineage)[:16]}"})
    return [field]


__all__ = ["build_sidra_demographic_population_fields"]
