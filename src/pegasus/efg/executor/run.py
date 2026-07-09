"""Physical tensor executor run loop for autonomous EFG.

Fixed-point driver that dispatches each field to its operator kernel. Pure code
motion from the former monolithic executor module.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.schemas import FieldNode
from pegasus.efg.dag import EFGResult

from pegasus.efg.executor.support import *
from pegasus.efg.executor.kernels import *


def _execute_non_rn(
    field: FieldNode, output_dir: Path, intent: Any = None, cache: "SourceScanCache | None" = None
) -> tuple[Path, int]:
    op = str(field.operator or "").lower()
    source = _source_path(field)

    # Only genuine pre-computed scalar anchors (e.g. the SIDRA population total)
    # use the scalar path. Count/aggregation fields carry incidental scalar keys
    # (n_events, count) in their support and MUST NOT short-circuit to a single
    # global value — they have to group by (year, municipality) over the source,
    # otherwise the Radon-Nikodym rate join finds no shared support axis.
    use_scalar = "anchor" in op or source is None
    if use_scalar:
        scalar = _scalar_tensor(field, output_dir)
        if scalar is not None:
            return scalar

    if source is None:
        raise ValueError("no source parquet path and no scalar support value")
    # Column resolution needs the source SCHEMA, not its data: take a 0-row head off the
    # cached lazy scan instead of re-reading 25 rows from disk per field.
    head = _scan_source(source, cache).limit(0).collect()
    column = _source_column(field, head)

    if op in {"count_measure", "count", "event_count"} or field.unit in {"counts", "count"}:
        out = _count_tensor(field, source, cache)
    elif column is not None and field.aggregation in {"additive", "statistical_functional", "weighted_mean"}:
        out = _sum_tensor(field, source, column, cache)
    elif column is not None:
        out = _source_field_tensor(field, source, column, cache)
    else:
        out = _count_tensor(field, source, cache)

    geo_mode = "native"
    if intent is not None:
        geo_mode = getattr(intent, "geo_mode", "native")
        if isinstance(intent, dict):
            geo_mode = intent.get("geo_mode", geo_mode)

    if geo_mode == "AMC" and "municipality_cod6" in out.columns:
        from pegasus.geo.amc import contract_to_amc
        # Look for the AMC crosswalk relative to the data lake root
        crosswalk_path = Path("data/raw/geo/amc_crosswalk.parquet")
        if crosswalk_path.exists():
            try:
                amc_result = contract_to_amc(out, crosswalk_path=str(crosswalk_path), value_column=VALUE_COLUMN, municipality_column="municipality_cod6")
                out = amc_result.frame.rename({"amc_id": "municipality_cod6"})
            except Exception as exc:
                # AMC harmonization was REQUESTED but failed — never silently emit native geography as
                # if it were AMC-contracted (report faithfully / never silently degrade). Surface it.
                warnings.warn(
                    f"AMC harmonization requested but contraction FAILED for field {field.id!r} "
                    f"({type(exc).__name__}: {exc}); emitting NATIVE geography for this field.",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            warnings.warn(
                f"AMC harmonization requested but the crosswalk is missing at {crosswalk_path}; "
                f"emitting NATIVE geography for field {field.id!r}.",
                RuntimeWarning, stacklevel=2,
            )

    return _write(output_dir / f"{field.id}.parquet", out)


def execute_efg_result(
    efg: EFGResult,
    *,
    output_dir: str | Path,
    require_materialized: bool = True,
    intent: Any = None,
) -> tuple[EFGResult, EFGExecutionReport]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scope_token = _GEO_SCOPE_PREFIXES.set(_scope_prefixes_from_intent(intent))
    try:
        return _execute_efg_result_impl(efg, out_dir=out_dir, require_materialized=require_materialized, intent=intent)
    finally:
        _GEO_SCOPE_PREFIXES.reset(scope_token)


def _execute_efg_result_impl(
    efg: EFGResult,
    *,
    out_dir: Path,
    require_materialized: bool,
    intent: Any = None,
) -> tuple[EFGResult, EFGExecutionReport]:
    fields_by_id: dict[str, FieldNode] = {field.id: field for field in efg.fields}
    executed: list[ExecutedField] = []

    # Finding 2: one lazy-scan cache per execution so N fields off the same events
    # parquet reuse a single scan plan instead of each re-opening the file. Stays lazy;
    # each field still terminates in its own streaming aggregate (bounded RAM).
    scan_cache = SourceScanCache()

    # Fixed-point execution: source/scalar fields first, then RN and bridges.
    pending = set(fields_by_id)
    last_error: dict[str, str] = {}
    for _ in range(max(2, len(fields_by_id) + 1)):
        progressed = False
        for field_id in list(pending):
            field = fields_by_id[field_id]
            op = str(field.operator or "")
            try:
                if op.upper() == "RN" or field.kind == "intensive_density":
                    path, rows, support_update = _compute_rn_ratio(field, fields_by_id, out_dir)
                elif op == "sidra_population_total_anchor":
                    path, rows = _sidra_population_tensor(field, out_dir)
                    support_update = None
                elif op == "sidra_context_field":
                    path, rows = _sidra_context_tensor(field, out_dir)
                    support_update = None
                elif op == "sidra_demographic_population":
                    path, rows = _sidra_demographic_population_tensor(field, out_dir)
                    support_update = None
                elif op == "psi_functional":
                    functional_source = _source_path(field)
                    if functional_source is None:
                        raise ValueError(f"functional field {field.id} has no source artifact")
                    path, rows = _write(out_dir / f"{field.id}.parquet", _functional_tensor(field, functional_source, scan_cache))
                    support_update = None
                elif op == "population_tensor_solver":
                    path, rows = _population_solver_tensor(field, out_dir)
                    support_update = None
                elif op.startswith("Bridge") or "bridge" in op.lower() or field.kind in {"bridge_module", "bridge_divergence"}:
                    path, rows, support_update = _compute_bridge_tensor(field, fields_by_id, out_dir)
                else:
                    path, rows = _execute_non_rn(field, out_dir, intent, scan_cache)
                    support_update = None
                if support_update:
                    field = field.model_copy(update={
                        "support": {**dict(field.support), **support_update},
                    })
                new_field = _materialized(field, path)
                fields_by_id[field_id] = new_field
                executed.append(ExecutedField(new_field, "success", str(path), rows))
                pending.remove(field_id)
                progressed = True
            except Exception as exc:
                # RN may be waiting for parent tensors. Keep it pending until the next pass,
                # but remember the real error so a permanently-blocked field reports WHY
                # instead of a generic "parents_not_materialized".
                last_error[field_id] = f"{type(exc).__name__}: {exc}"
                # RN and cross-source bridges/divergences depend on parent tensors that may
                # not be materialized yet; keep them pending across fixed-point passes.
                if (
                    op.upper() == "RN"
                    or field.kind in {"intensive_density", "bridge_divergence", "bridge_module"}
                    or op.startswith("Bridge")
                    or op == "divergence_log_ratio"
                ):
                    continue
                executed.append(ExecutedField(field, "blocked", None, 0, last_error[field_id]))
                pending.remove(field_id)
                progressed = True
        if not pending or not progressed:
            break

    for field_id in sorted(pending):
        field = fields_by_id[field_id]
        reason = last_error.get(field_id, "parents_not_materialized_or_operator_not_executable")
        executed.append(ExecutedField(field, "blocked", None, 0, reason))

    blocked = [item for item in executed if item.status != "success"]
    if require_materialized and blocked:
        msg = "; ".join(f"{item.field.name}:{item.reason}" for item in blocked[:40])
        raise RuntimeError(f"Autonomous EFG physical execution blocked {len(blocked)} fields: {msg}")

    new_efg = EFGResult(
        schema_version=efg.schema_version,
        efg_id=efg.efg_id,
        substrate_id=efg.substrate_id,
        fields=tuple(fields_by_id[field.id] for field in efg.fields),
        edges=efg.edges,
        failed_branches=efg.failed_branches,
        warnings=efg.warnings,
        variable_dictionary=efg.variable_dictionary,
        precompression=efg.precompression,
        source_hashes=efg.source_hashes,
        registry_hashes=efg.registry_hashes,
        legality_summary=efg.legality_summary,
        operator_mode=efg.operator_mode,
        core_seed_summary=efg.core_seed_summary,
        bridge_plan_summary=efg.bridge_plan_summary,
        domain_summaries=efg.domain_summaries,
    )
    report = EFGExecutionReport(
        status="success" if not blocked else "blocked",
        executed_count=sum(1 for item in executed if item.status == "success"),
        blocked_count=len(blocked),
        fields=tuple(executed),
        output_dir=str(out_dir),
    )
    return new_efg, report


__all__ = [
    "_execute_non_rn",
    "execute_efg_result",
    "_execute_efg_result_impl",
]
