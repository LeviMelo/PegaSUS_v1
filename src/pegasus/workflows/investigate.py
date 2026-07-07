"""Investigate workflow — compile → CommonPanel → LDO → LinkRecords (MSD-II §II.6, MII-OUT-01).

The single ExecutionStage=investigate entrypoint that runs the Lattice Dependency
Operator over a compiled run's CommonPanel and writes typed link records to the
Hypotheses key. This replaces the PIRS slice-zoo's design/selection/candidate/
scan chain with one in-memory orchestrator call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.ldo.field_selection import analytical_variable_ids
from pegasus.ldo.orchestrator import run_ldo
from pegasus.ldo.output import write_hypotheses
from pegasus.she.panel import Resolution, compile_common_panel

# support.restrict_conditions ops whose ``value`` enumerates the codes a σ_C count
# variable is built on (a shared-code overlap between two such variables is mechanical).
_ICD_ENUM_OPS: frozenset[str] = frozenset({"starts_with_any", "is_in", "in", "eq", "equals"})


@dataclass
class InvestigateResult:
    run_dir: str
    n_link_records: int
    n_selected: int
    hypotheses_path: str
    panel_cells: int
    panel_fields: int
    diagnostics: dict[str, Any]


def _geography_prefixes(intent: Any) -> frozenset[str] | None:
    from pegasus.efg.executor import _scope_prefixes_from_intent

    return _scope_prefixes_from_intent(intent)


def _loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _icd_code_set(support: dict[str, Any]) -> frozenset[str] | None:
    """The ICD codes a σ_C restriction count is built on (from ``restrict_conditions``).

    A cause-specific count field carries a predicate like
    ``{"column": "principal_icd_norm", "op": "starts_with_any", "value": ["A90", "A91"]}``;
    the enumerated ``value`` IS the variable's code set (§5.3). Returns None for fields
    that carry no ICD-enumerating predicate (non-disease variables — left untouched).
    """
    conds = support.get("restrict_conditions")
    if not isinstance(conds, list):
        return None
    codes: set[str] = set()
    for cond in conds:
        if not isinstance(cond, dict):
            continue
        column = str(cond.get("column") or "").lower()
        if "icd" not in column and "cid" not in column:
            continue
        if str(cond.get("op") or "") not in _ICD_ENUM_OPS:
            continue
        raw = cond.get("value")
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        codes.update(str(v).strip().upper() for v in values if v is not None and str(v).strip())
    return frozenset(codes) or None


def disease_variable_meta(
    run_dir: str | Path, keep_variables: set[str] | frozenset[str] | None = None
) -> dict[str, dict]:
    """Build ``run_ldo(variable_meta=...)`` for a compiled run from its ``V_fields`` metadata.

    The disease-coded LDO variables are the EFG's σ_C restriction counts, whose ICD code
    set is persisted per field in ``support.restrict_conditions`` and whose SIM topology
    role is persisted in ``axes.icd_topology_role``. This reads those directly — no
    regeneration through the disease variable grammar — so only fields that genuinely carry
    codes appear in the map. Non-disease runs yield an empty map (LDO behaviour unchanged).
    """
    path = Path(run_dir) / "V_fields.parquet"
    if not path.exists():
        return {}
    vf = pl.read_parquet(path)
    if "field_id" not in vf.columns:
        return {}
    cols = [c for c in ("field_id", "support_json", "axes_json") if c in vf.columns]
    meta: dict[str, dict] = {}
    for row in vf.select(cols).iter_rows(named=True):
        fid = str(row["field_id"])
        if keep_variables is not None and fid not in keep_variables:
            continue
        support = _loads(row.get("support_json")) or {}
        axes = _loads(row.get("axes_json")) or {}
        code_set = _icd_code_set(support) if isinstance(support, dict) else None
        topology = axes.get("icd_topology_role") or axes.get("diagnostic_role") if isinstance(axes, dict) else None
        if code_set is None and topology is None:
            continue
        entry: dict[str, Any] = {"code_system": "CID-10"}
        if code_set is not None:
            entry["code_set"] = code_set
        if topology is not None:
            entry["topology_role"] = str(topology)
        meta[fid] = entry
    return meta


def measured_quantity_refs(run_dir, keep_variables=None) -> dict[str, str]:
    """``{field_id: sidecar_path}`` for RN fields that emitted a §II.3 MeasuredQuantity sidecar.

    Read from ``V_fields`` (a ``measured_quantity_ref`` column or the field's ``support_json``).
    Enables the §III.5 count-with-exposure margin on real runs: the LDO consumes the raw count
    + exposure from the sidecar instead of the pre-divided rate. Empty when no RN field emitted
    one (LDO behaviour unchanged)."""
    path = Path(run_dir) / "V_fields.parquet"
    if not path.exists():
        return {}
    vf = pl.read_parquet(path)
    if "field_id" not in vf.columns:
        return {}
    cols = ["field_id"]
    if "measured_quantity_ref" in vf.columns:
        cols.append("measured_quantity_ref")
    if "support_json" in vf.columns:
        cols.append("support_json")
    out: dict[str, str] = {}
    for row in vf.select(cols).iter_rows(named=True):
        fid = str(row["field_id"])
        if keep_variables is not None and fid not in keep_variables:
            continue
        ref = row.get("measured_quantity_ref")
        if not ref and "support_json" in row:
            support = _loads(row.get("support_json")) or {}
            ref = support.get("measured_quantity_ref") if isinstance(support, dict) else None
        if not ref:
            continue
        p = Path(ref)
        if not p.is_absolute():
            p = Path(run_dir) / p
        if p.exists():
            out[fid] = str(p)
    return out


def _disease_graph_from_meta(variable_meta: dict[str, dict]):
    """A variable-keyed structural DiseaseGraph over the code-bearing variables, or None.

    Keyed by variable id (not bare ICD codes) so it actually intersects the LDO variable
    set inside ``disease_penalty_matrix`` — a code-keyed graph would be a silent no-op.
    """
    code_sets = {
        v: m["code_set"] for v, m in variable_meta.items() if m.get("code_set")
    }
    if len(code_sets) < 2:
        return None
    from pegasus.disease.graph import DiseaseGraph

    graph = DiseaseGraph.from_variable_code_sets(code_sets)
    # Only worth threading if it carries at least one structural coupling; an all-zero
    # graph would leave the penalty at the scalar baseline anyway.
    return graph if graph.weights.nnz > 0 else None


def run_investigate(
    run_dir: str | Path,
    *,
    intent: Any = None,
    resolution: Resolution = "year",
    K: int = 3,
    lambda1: float = 0.1,
    lambda2: float = 1.0,
    write: bool = True,
    **ldo_kwargs,
) -> InvestigateResult:
    """Run the LDO over a compiled run and (optionally) write the Hypotheses key.

    Production defaults: ``K=3`` (adaptively capped by the panel's time span / size,
    see ``run_ldo``) and ``lambda2=1.0`` — the raw ``run_ldo`` default of 0.1 lets the
    low-rank layer over-absorb on noisy real panels (nearly full rank, collapsing
    every relationship into latent_shared), so a stronger nuclear-norm penalty is the
    right production default; callers may override.
    """
    run_dir = Path(run_dir)
    prefixes = _geography_prefixes(intent)
    panel = compile_common_panel(run_dir, resolution=resolution, geography_prefixes=prefixes)

    # Restrict the LDO to analytical fields (drop raw source-column passthroughs and
    # support axes) unless the caller overrides — smaller precision solve, cleaner graph.
    keep_variables = ldo_kwargs.pop("keep_variables", None)
    if keep_variables is None:
        keep_variables = analytical_variable_ids(run_dir)

    # Disease-axis wiring (refactor §4.1): construct the variable_meta (code_set +
    # SIM topology role, from V_fields) and the structural DiseaseGraph (over the union
    # of variable code sets) so the DIS-04 L_D prior, the §III.8 mechanical-overlap guard,
    # and disease provenance annotation are LIVE on real runs — not inert as before, when
    # run_ldo was called with these left None. A run with no disease-coded variables yields
    # an empty meta / None graph, so non-disease runs are unaffected (no regression).
    variable_meta = ldo_kwargs.pop("variable_meta", None)
    if variable_meta is None:
        variable_meta = disease_variable_meta(run_dir, keep_variables) or None
    disease_graph = ldo_kwargs.pop("disease_graph", None)
    if disease_graph is None and variable_meta is not None:
        disease_graph = _disease_graph_from_meta(variable_meta)

    # §III.5 count-with-exposure: hand the LDO the RN fields' MeasuredQuantity sidecars so it
    # models raw count + exposure (offset) rather than the pre-divided rate. Empty on runs
    # without RN sidecars → no behaviour change.
    mq_by_var = ldo_kwargs.pop("measured_quantity_by_variable", None)
    if mq_by_var is None:
        mq_by_var = measured_quantity_refs(run_dir, keep_variables) or None

    # §III.3/§III.7 spatial BYM field: persist per-edge spatial-heterogeneity surfaces under the
    # run dir so certified edges carry a spatial_field_ref (effect-modification by place).
    spatial_field_dir = ldo_kwargs.pop("spatial_field_dir", None)
    if spatial_field_dir is None:
        spatial_field_dir = str(Path(run_dir) / "spatial_fields")

    ldo_run = run_ldo(
        panel, K=K, lambda1=lambda1, lambda2=lambda2, keep_variables=keep_variables,
        variable_meta=variable_meta, disease_graph=disease_graph,
        measured_quantity_by_variable=mq_by_var, spatial_field_dir=spatial_field_dir,
        **ldo_kwargs,
    )

    # Record the disease-axis wiring so a run's diagnostics show whether the L_D prior /
    # overlap guard were live (rather than the wiring being silently inert as before).
    ldo_run.diagnostics["disease_variables"] = 0 if not variable_meta else sum(
        1 for m in variable_meta.values() if m.get("code_set")
    )
    ldo_run.diagnostics["disease_graph_applied"] = disease_graph is not None

    hypotheses_path = run_dir / "Hypotheses.parquet"
    if write:
        write_hypotheses(ldo_run.link_records, hypotheses_path)

    return InvestigateResult(
        run_dir=str(run_dir),
        n_link_records=len(ldo_run.link_records),
        n_selected=ldo_run.diagnostics.get("n_selected", 0),
        hypotheses_path=str(hypotheses_path),
        panel_cells=panel.index.height,
        panel_fields=len(panel.fields),
        diagnostics=ldo_run.diagnostics,
    )


__all__ = ["InvestigateResult", "run_investigate", "disease_variable_meta"]
