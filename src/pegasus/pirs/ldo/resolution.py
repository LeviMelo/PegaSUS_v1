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

from pegasus.pirs.ldo.margins import GaussianField
from pegasus.pirs.ldo.orchestrator import LDORun, run_ldo
from pegasus.pirs.ldo.records import LinkRecord


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


def _candidate_pairs(run: LDORun) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for r in run.link_records:
        if r.certification_status != "selected":
            continue
        if r.edge_type not in {"lagged_directed", "contemporaneous", "nonlinear_residual", "latent_shared"}:
            continue
        key = (r.source_var, r.target_var)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def run_multiresolution_ldo(
    coarse_field: GaussianField,
    fine_field: GaussianField,
    *,
    coarse_K: int = 4,
    fine_K: int = 12,
    seed: int = 0,
    **ldo_kwargs,
) -> MultiResolutionRun:
    """Coarse discovery then fine refinement of candidate edges."""
    coarse = run_ldo(coarse_field, K=coarse_K, seed=seed, **ldo_kwargs)
    candidates = _candidate_pairs(coarse)

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

    return MultiResolutionRun(
        coarse=coarse,
        fine=fine,
        link_records=merged,
        candidates=candidates,
        diagnostics={
            "candidates": len(candidates),
            "candidate_vars": list(candidate_vars),
            "coarse_selected": coarse.diagnostics.get("n_selected"),
            "fine_selected": fine.diagnostics.get("n_selected"),
        },
    )


__all__ = ["MultiResolutionRun", "run_multiresolution_ldo", "restrict_variables"]
