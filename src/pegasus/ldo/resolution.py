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


_LINK_EDGE_TYPES = {"lagged_directed", "contemporaneous", "nonlinear_residual", "latent_shared"}


def _sensitivity_screen(run: LDORun, *, sensitivity_threshold: float = 0.05) -> list[tuple[str, str]]:
    """§VIII.2(1): screen on SENSITIVITY, not the aggregate-certified TEST.

    The coarse pass is a recall-tuned FILTER, not a test: drill down on any pair with a
    non-trivial signal — certified ``selected`` OR ``|weight|`` above a low threshold OR any
    stability recurrence — so a sub-threshold-but-real edge (e.g. a cancellation edge whose
    pooled effect is near zero but whose subgroup signal is strong) is not pruned before the
    fine pass can look. Using the certified test here would silently drop exactly those.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in run.link_records:
        if r.edge_type not in _LINK_EDGE_TYPES:
            continue
        sensitive = (
            r.certification_status == "selected"
            or abs(r.weight or 0.0) >= sensitivity_threshold
            or (r.stability or 0.0) > 0.0
        )
        if sensitive:
            key = (r.source_var, r.target_var)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


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
    candidates = _sensitivity_screen(coarse, sensitivity_threshold=sensitivity_threshold)  # §VIII.2(1)
    coarse_vars = list(coarse_field.variables)

    if not candidates:
        return MultiResolutionRun(
            coarse=coarse, fine=None, link_records=coarse.link_records, candidates=[],
            diagnostics={"candidates": 0, "reason": "no_coarse_candidates"},
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

    return MultiResolutionRun(
        coarse=coarse,
        fine=fine,
        link_records=merged,
        candidates=candidates,
        diagnostics={
            "candidates": len(candidates),
            "candidate_vars": list(candidate_vars),
            "n_pruned_vars": len(pruned_vars),
            "coarse_selected": coarse.diagnostics.get("n_selected"),
            "fine_selected": fine.diagnostics.get("n_selected"),
            # §VIII.2(1) recall-tuned screen + §VIII.2(2) empirical false-negative measurement
            "sensitivity_screen": "recall_tuned_signal_or_stability",
            "random_deep_audit": audit,
        },
    )


__all__ = ["MultiResolutionRun", "run_multiresolution_ldo", "restrict_variables"]
