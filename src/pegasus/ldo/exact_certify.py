"""Exact-certifies-approximate cross-scale validation (MSD-III §V.6(3), §IX.3).

A national run uses approximations (randomized SVD, matrix-free whitening, coarsened HSIC);
a state-scale EXACT run does not. Where they overlap (the edges whose endpoint variables are
both present in a state slice), the exact result MUST agree with the approximate one within
the propagated uncertainty band — disagreement beyond the band rejects the approximation
LOUDLY (never silently). This is the normative validity contract for the approximations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pegasus.ldo.records import LinkRecord


class ApproximationRejectedError(RuntimeError):
    """Raised (in strict mode) when the exact run disagrees with the approximate beyond bounds."""


@dataclass(frozen=True)
class EdgeDisagreement:
    source_var: str
    target_var: str
    lag_k: int
    approx_weight: float
    exact_weight: float
    band: float


def _key(r: LinkRecord) -> tuple[frozenset, int, str]:
    return (frozenset((r.source_var, r.target_var)), r.lag_k, r.edge_type)


def certify_exact_vs_approx(
    approx_records: list[LinkRecord],
    exact_records: list[LinkRecord],
    *,
    tolerance_sigma: float = 3.0,
    strict: bool = False,
) -> dict[str, Any]:
    """Compare overlapping edges of an APPROXIMATE (national) vs EXACT (state-scale) run.

    Two edges overlap when they share endpoints, lag, and type. They AGREE when the absolute
    weight difference is within ``tolerance_sigma × √(σ_approx² + σ_exact²)`` (the propagated
    band from each edge's uncertainty). Returns a report; ``strict=True`` raises
    :class:`ApproximationRejectedError` on any disagreement (the §V.6(3) "loud rejection").
    """
    a = {_key(r): r for r in approx_records}
    e = {_key(r): r for r in exact_records}
    overlap = sorted(set(a) & set(e), key=lambda k: sorted(k[0]))
    disagreements: list[EdgeDisagreement] = []
    for k in overlap:
        ra, re = a[k], e[k]
        sa = ra.uncertainty if ra.uncertainty is not None else 0.0
        se = re.uncertainty if re.uncertainty is not None else 0.0
        band = tolerance_sigma * math.hypot(sa, se)
        diff = abs((ra.weight or 0.0) - (re.weight or 0.0))
        if diff > band:  # band==0 (no uncertainty) → any nonzero diff disagrees
            src, tgt = sorted(k[0]) if len(k[0]) == 2 else (ra.source_var, ra.target_var)
            disagreements.append(EdgeDisagreement(
                source_var=src, target_var=tgt, lag_k=ra.lag_k,
                approx_weight=float(ra.weight or 0.0), exact_weight=float(re.weight or 0.0), band=float(band),
            ))
    report = {
        "n_overlap": len(overlap),
        "n_disagree": len(disagreements),
        "certified": len(disagreements) == 0,
        "tolerance_sigma": tolerance_sigma,
        "disagreements": [
            {"source_var": d.source_var, "target_var": d.target_var, "lag_k": d.lag_k,
             "approx_weight": d.approx_weight, "exact_weight": d.exact_weight, "band": d.band}
            for d in disagreements
        ],
    }
    if strict and disagreements:
        raise ApproximationRejectedError(
            f"exact_certifies_approximate FAILED: {len(disagreements)}/{len(overlap)} overlapping "
            f"edges disagree beyond {tolerance_sigma}σ propagated bounds — the national "
            "approximation is rejected (§V.6(3))."
        )
    return report


__all__ = ["certify_exact_vs_approx", "ApproximationRejectedError", "EdgeDisagreement"]
