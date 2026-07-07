"""Coarse→fine multi-resolution LDO (MSD-II §II.7, MII-RES-01).

Resolution is a first-class axis. The LDO runs coarse→fine:
- **Coarse pass** (small ``S·T``, e.g. municipality×year or region×month): fit the
  global link graph and take the selected edges as *candidates*.
- **Fine pass** (municipality×month): refit *only the candidate variables* to
  sharpen the lag peak and localise the effect.

This is simultaneously the scientific multi-resolution story and the laptop
compute strategy (each pass is envelope-sized). The two passes must be
*consistent* — a candidate edge present coarse should survive fine (possibly at a
sharper lag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pegasus.ldo.exhaustiveness import CoverageManifest, should_drill_down
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.orchestrator import LDORun, run_ldo
from pegasus.ldo.records import LinkRecord


@dataclass
class MultiResolutionRun:
    coarse: LDORun
    fine: LDORun | None
    link_records: list[LinkRecord]
    candidates: list[tuple[str, str]]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    coverage_manifest: CoverageManifest | None = None    # §VIII.2 typed searched/unsearched record


def restrict_variables(field: GaussianField, variables: tuple[str, ...]) -> GaussianField:
    """Subset a GaussianField to the given variables (order preserved)."""
    keep = [i for i, v in enumerate(field.variables) if v in set(variables)]
    return GaussianField(
        variables=tuple(field.variables[i] for i in keep),
        space_ids=field.space_ids,
        time_ids=field.time_ids,
        Z=field.Z[keep],
        W=field.W[keep],
        resolution=field.resolution,
    )


def coarsen_field_spatial(field: GaussianField, *, level: int = 2) -> GaussianField:
    """Aggregate a GaussianField to a coarser spatial grain by cod6 prefix (§II.7 coarse pass).

    Municipalities sharing the first ``level`` cod6 digits (``level=2`` → UF/state, ``level=1``
    → macroregion) are pooled: each coarse cell is the reliability-weighted mean of its member
    municipalities' values, per variable and time. The coarse field has far fewer spatial units
    (``S≈27`` at UF) so the coarse discovery pass is cheap and envelope-sized, while the fine
    field stays at municipality grain for candidate refinement. Cells with no weight stay NaN.
    """
    ids = [str(s) for s in field.space_ids]
    prefixes = [s[:level] if len(s) >= level else s for s in ids]
    groups = sorted(set(prefixes))
    gpos = {g: i for i, g in enumerate(groups)}
    member_idx: list[list[int]] = [[] for _ in groups]
    for i, pre in enumerate(prefixes):
        member_idx[gpos[pre]].append(i)

    p, _, T = field.Z.shape
    G = len(groups)
    Zc = np.full((p, G, T), np.nan, dtype=np.float64)
    Wc = np.zeros((p, G, T), dtype=np.float64)
    W = field.W if getattr(field, "W", None) is not None else np.ones_like(field.Z)
    for g, members in enumerate(member_idx):
        idx = np.asarray(members, dtype=np.int64)
        w = np.where(np.isfinite(field.Z[:, idx, :]), W[:, idx, :], 0.0)  # (p, |members|, T)
        vals = np.where(np.isfinite(field.Z[:, idx, :]), field.Z[:, idx, :], 0.0)
        wsum = w.sum(axis=1)                                             # (p, T)
        num = (w * vals).sum(axis=1)                                     # (p, T)
        ok = wsum > 0
        Zc[:, g, :] = np.where(ok, num / np.where(ok, wsum, 1.0), np.nan)
        Wc[:, g, :] = np.where(ok, wsum / np.maximum(1, len(members)), 0.0)  # mean reliability
    return GaussianField(
        variables=field.variables, space_ids=tuple(groups), time_ids=field.time_ids,
        Z=Zc, W=Wc, resolution=f"{field.resolution}_L{level}",
    )


_LINK_EDGE_TYPES = {"lagged_directed", "contemporaneous", "nonlinear_residual", "latent_shared"}


def _subgroup_effect_vectors(field: GaussianField) -> np.ndarray:
    """Per-spatial-unit within-unit effect of every variable pair — the ``(p, p, S)`` tensor
    whose ``[i, j, :]`` slice is the subgroup-effect vector for pair ``(i, j)`` (§VIII.2(1)).

    The effect in unit ``s`` is the within-unit correlation of variables ``i`` and ``j`` over
    time. A pair whose per-unit effects have opposite signs (Simpson/cancellation) has a pooled
    mean ≈ 0 but a high spread here; a pair strong in one unit only (sparsity dilution) has a
    high max here. These are exactly the signals the pooled aggregate hides.
    """
    Z = np.asarray(field.Z, dtype=np.float64)      # (p, S, T)
    p, S, T = Z.shape
    Zc = np.where(np.isfinite(Z), Z, np.nan)
    mean = np.nanmean(Zc, axis=2, keepdims=True)
    std = np.nanstd(Zc, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        Zs = (Zc - mean) / np.where(std > 1e-9, std, np.nan)
    Zs = np.where(np.isfinite(Zs), Zs, 0.0)         # unit/var with no variance → 0 contribution
    eff = np.einsum("ist,jst->ijs", Zs, Zs) / max(T - 1, 1)   # (p, p, S) per-unit correlations
    return eff


def _sensitivity_screen(
    run: LDORun, field: GaussianField, *,
    het_threshold: float = 0.15, max_threshold: float = 0.30, weight_floor: float = 0.05,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str]]:
    """§VIII.2(1): screen on SENSITIVITY, not the aggregate-certified TEST.

    The coarse pass is a recall-tuned FILTER, not a test. A pair is drilled to fine grain when
    EITHER (a) its per-spatial-unit effect vector fires ``should_drill_down`` — subgroup
    heterogeneity (cancellation) OR max-subgroup (localized) signal, the statistics designed to
    fire on what aggregation hides — OR (b) it already carries a pooled coarse signal (certified
    ``selected`` / ``|weight| ≥ floor`` / any stability recurrence). The subgroup limb is the fix
    for the anticonservative pooled-mean gate: a cancellation edge whose pooled effect ≈ 0 is now
    kept. Returns ``(pairs, reason_by_pair)``.
    """
    variables = list(field.variables)
    idx = {v: i for i, v in enumerate(variables)}
    eff = _subgroup_effect_vectors(field) if field.Z.shape[1] > 1 else None
    pairs: list[tuple[str, str]] = []
    reasons: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str]] = set()

    # (a) subgroup-heterogeneity / max-subgroup drill over EVERY pair (the recall filter).
    if eff is not None:
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                drill, reason = should_drill_down(
                    eff[i, j], het_threshold=het_threshold, max_threshold=max_threshold)
                if drill:
                    key = (variables[i], variables[j])
                    seen.add(key)
                    pairs.append(key)
                    reasons[key] = f"subgroup:{reason}"

    # (b) union with any pooled coarse signal, so a clearly-selected edge is never dropped.
    for r in run.link_records:
        if r.edge_type not in _LINK_EDGE_TYPES:
            continue
        pooled = (
            r.certification_status == "selected"
            or abs(r.weight or 0.0) >= weight_floor
            or (r.stability or 0.0) > 0.0
        )
        if pooled:
            key = (r.source_var, r.target_var)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
                reasons[key] = "pooled_coarse_signal"
    return pairs, reasons


def _random_deep_audit(
    fine_field: GaussianField, *, pruned_vars: list[str], fine_K: int, seed: int,
    n_audit_vars: int, **ldo_kwargs,
) -> dict[str, Any]:
    """§VIII.2(2): fine-analyze a random sample of PRUNED variables to MEASURE the coarse
    screen's false-negative rate empirically (a pruned pair that IS significant at fine grain
    is a missed discovery). Reported, never hidden — so the sparsity assumption is checked."""
    if len(pruned_vars) < 2 or n_audit_vars < 2:
        return {"n_audit_vars": 0, "audited_selected_edges": 0, "false_negative_rate": None}
    rng = np.random.default_rng(seed)
    k = min(n_audit_vars, len(pruned_vars))
    sample = [pruned_vars[i] for i in sorted(rng.choice(len(pruned_vars), size=k, replace=False))]
    audit_field = restrict_variables(fine_field, tuple(sample))
    try:
        audit = run_ldo(audit_field, K=fine_K, seed=seed + 7, run_residual_scan=False, **ldo_kwargs)
    except Exception as exc:  # audit is diagnostic; never fail the run
        return {"n_audit_vars": k, "audited_selected_edges": None, "false_negative_rate": None,
                "audit_error": f"{type(exc).__name__}"}
    n_selected = sum(1 for r in audit.link_records if r.certification_status == "selected")
    n_pairs = max(1, k * (k - 1) // 2)
    return {
        "n_audit_vars": k,
        "audited_selected_edges": n_selected,
        # fraction of audited pruned pairs that turned up significant at fine grain
        "false_negative_rate": float(n_selected) / float(n_pairs),
    }


def run_multiresolution_ldo(
    coarse_field: GaussianField,
    fine_field: GaussianField,
    *,
    coarse_K: int = 4,
    fine_K: int = 12,
    seed: int = 0,
    sensitivity_threshold: float = 0.05,
    n_audit_vars: int = 8,
    **ldo_kwargs,
) -> MultiResolutionRun:
    """Coarse discovery then fine refinement of candidate edges (§II.7), with the §VIII.2
    bounded-exhaustiveness remedy: a sensitivity screen (not the certified test) selects
    candidates, and a random deep audit of pruned variables measures the false-negative rate."""
    coarse = run_ldo(coarse_field, K=coarse_K, seed=seed, **ldo_kwargs)
    candidates, reasons = _sensitivity_screen(coarse, coarse_field)  # §VIII.2(1) subgroup screen
    coarse_vars = list(coarse_field.variables)
    coarse_res = str(getattr(coarse_field, "resolution", "coarse"))
    fine_res = str(getattr(fine_field, "resolution", "fine"))

    # §VIII.2(3) typed coverage manifest: every drilled pair is a SEARCHED region (with the
    # sensitivity reason it fired); every screened-out pair is an explicit UNSEARCHED region
    # (with the reason it was pruned) — never a silent gap. The sparsity-of-truth assumption is
    # a STATED string on the manifest, not a hidden bool.
    manifest = CoverageManifest()
    cand_set = set(candidates)
    for i, a in enumerate(coarse_vars):
        for b in coarse_vars[i + 1:]:
            key = (a, b)
            if key in cand_set:
                manifest.mark_searched(region=f"{a}~{b}", resolution=fine_res)
            else:
                manifest.mark_unsearched(region=f"{a}~{b}", resolution=fine_res,
                                         reason="below_sensitivity_thresholds(subgroup+pooled)")

    if not candidates:
        return MultiResolutionRun(
            coarse=coarse, fine=None, link_records=coarse.link_records, candidates=[],
            diagnostics={"candidates": 0, "reason": "no_coarse_candidates",
                         "sensitivity_screen": "subgroup_heterogeneity_or_max_or_pooled"},
            coverage_manifest=manifest,
        )

    candidate_vars = tuple(dict.fromkeys([v for pair in candidates for v in pair]))
    fine_sub = restrict_variables(fine_field, candidate_vars)
    fine = run_ldo(fine_sub, K=fine_K, seed=seed + 100, **ldo_kwargs)

    # Merge: fine records for candidate pairs (sharper), plus coarse records for
    # any pair the fine pass did not re-examine.
    fine_pairs = {(r.source_var, r.target_var) for r in fine.link_records}
    merged: list[LinkRecord] = list(fine.link_records)
    for r in coarse.link_records:
        if (r.source_var, r.target_var) not in fine_pairs:
            merged.append(r)

    # §VIII.2(2): random deep audit of the PRUNED variables — measure the false-negative rate.
    pruned_vars = [v for v in coarse_vars if v not in set(candidate_vars)]
    audit = _random_deep_audit(
        fine_field, pruned_vars=pruned_vars, fine_K=fine_K, seed=seed + 200,
        n_audit_vars=n_audit_vars, **ldo_kwargs,
    )
    manifest.random_audit_false_negative_rate = audit.get("false_negative_rate")

    return MultiResolutionRun(
        coarse=coarse,
        fine=fine,
        link_records=merged,
        candidates=candidates,
        diagnostics={
            "candidates": len(candidates),
            "candidate_vars": list(candidate_vars),
            "candidate_reasons": {f"{a}~{b}": reasons.get((a, b)) for (a, b) in candidates},
            "n_pruned_vars": len(pruned_vars),
            "coarse_selected": coarse.diagnostics.get("n_selected"),
            "fine_selected": fine.diagnostics.get("n_selected"),
            # §VIII.2(1) subgroup sensitivity screen + §VIII.2(2) empirical false-negative measurement
            "sensitivity_screen": "subgroup_heterogeneity_or_max_or_pooled",
            "random_deep_audit": audit,
        },
        coverage_manifest=manifest,
    )


__all__ = ["MultiResolutionRun", "run_multiresolution_ldo", "restrict_variables", "coarsen_field_spatial"]
