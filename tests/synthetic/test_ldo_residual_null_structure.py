"""LDO residual-HSIC structured-null validity (§6.8 anticonservativeness fix).

Residuals over a spatial panel retain spatial autocorrelation, so an i.i.d.
permutation null is *anticonservative*: two variables that merely co-cluster in
space (both elevated in the same states) test as dependent. The residual scan must
use a within-spatial-block (UF) restricted permutation, whose null preserves the
shared spatial clustering — so a spurious "both high in the Northeast" pair is NOT
certified as a nonlinear edge.

This test plants two conditionally-independent-given-UF variables with a strong
shared spatial mean and asserts the structured null does not certify the spurious
edge, while the iid fallback (single spatial block) does — the concrete
anticonservativeness the fix removes.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.margins import GaussianField
from pegasus.ldo.residual_scan import scan_residual_nonlinear_edges


def _clustered_independent_field(*, single_block: bool):
    """p=3 variables over 60 munis (across 6 UFs) x 4 years. Vars 0 and 1 share a
    strong UF-level mean but are otherwise independent (independent given UF)."""
    rng = np.random.default_rng(11)
    n_uf, per_uf, T = 6, 10, 4
    S = n_uf * per_uf
    uf_of_muni = np.repeat(np.arange(n_uf), per_uf)
    uf_effect0 = rng.standard_normal(n_uf) * 2.0
    uf_effect1 = uf_effect0 * 1.8  # vars 0 and 1 share the SAME spatial pattern
    Z = np.empty((3, S, T))
    for s in range(S):
        u = uf_of_muni[s]
        Z[0, s, :] = uf_effect0[u] + rng.standard_normal(T) * 0.4
        Z[1, s, :] = uf_effect1[u] + rng.standard_normal(T) * 0.4  # independent noise, shared UF mean
        Z[2, s, :] = rng.standard_normal(T)
    if single_block:
        space_ids = tuple(f"27{i:05d}"[:7] for i in range(S))  # all UF "27" → iid fallback
    else:
        space_ids = tuple(f"{27 + uf_of_muni[s]}{s:05d}"[:7] for s in range(S))  # 6 distinct UFs
    return GaussianField(
        variables=("A", "B", "N"),
        space_ids=space_ids,
        time_ids=tuple(range(T)),
        Z=Z,
        W=np.ones_like(Z),
        resolution="year",
    )


def _all_pairs(records):
    return {tuple(sorted((r.source_var, r.target_var))) for r in records}


def test_structured_null_suppresses_spurious_spatially_clustered_edge():
    field = _clustered_independent_field(single_block=False)
    precision = np.eye(3)  # residuals ≈ standardized fields
    records = scan_residual_nonlinear_edges(field, precision, permutations=300, seed=0)
    assert all(r.null_strategy == "restricted_intra_uf_spatial_swap" for r in records), \
        f"expected structured null, got {[r.null_strategy for r in records]}"
    # the spurious A-B pair (independent given UF, only co-clustered in space) is not even
    # significant under the within-UF null — its shared spatial mean is preserved in the null
    assert ("A", "B") not in _all_pairs(records), \
        "structured null still flagged the spatially-confounded A-B edge (anticonservative)"


def test_iid_fallback_flags_spurious_edge_and_gate_blocks_certification():
    """Collapsing every muni into one UF forces the iid fallback: the same spurious A-B
    co-clustering now *is* found significant (anticonservative) — but the insufficient-
    spatial-block gate refuses to certify it as `selected`. Both guards are exercised."""
    field = _clustered_independent_field(single_block=True)
    precision = np.eye(3)
    records = scan_residual_nonlinear_edges(field, precision, permutations=300, seed=0)
    assert all(r.null_strategy == "iid_permutation" for r in records)
    # iid null finds the spatially-confounded edge "significant" (the anticonservativeness)
    assert ("A", "B") in _all_pairs(records)
    ab = next(r for r in records if tuple(sorted((r.source_var, r.target_var))) == ("A", "B"))
    # ...but a single spatial block is an invalid null, so it is not certified selected
    assert ab.certification_status == "descriptive", \
        "single-block iid null must not certify a discovery as selected"
