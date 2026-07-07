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
    # §6.8 panel-aware null: an annual municipal panel selects the spatial-block × time-bucket
    # structured swap, never iid.
    assert all(r.null_strategy != "iid_permutation" for r in records), \
        f"expected structured null, got {[r.null_strategy for r in records]}"
    # the spurious A-B pair (independent given UF, only co-clustered in space) is not even
    # significant under the within-block null — its shared spatial mean is preserved in the null
    assert ("A", "B") not in _all_pairs(records), \
        "structured null still flagged the spatially-confounded A-B edge (anticonservative)"


def test_single_spatial_block_uses_temporal_null_but_gate_blocks_certification():
    """One (mislabeled) spatial block over multiple years: there is still TEMPORAL structure,
    so the §6.8 selection uses the panel's structured null (not iid) — a strict improvement over
    the old spatial-only fallback. The spatial blocks are invalid for the spatial confound, so
    the same spurious A-B pair is found but the disjunctive block gate (< 5 spatial OR temporal
    blocks) refuses to certify it `selected`."""
    field = _clustered_independent_field(single_block=True)  # T=4 → temporal blocks exist
    precision = np.eye(3)
    records = scan_residual_nonlinear_edges(field, precision, permutations=300, seed=0)
    assert records and all(r.null_strategy != "iid_permutation" for r in records), \
        "with temporal structure the null must be the panel-aware structured null, not iid"
    assert ("A", "B") in _all_pairs(records)  # spatial confound not broken (blocks invalid)
    ab = next(r for r in records if tuple(sorted((r.source_var, r.target_var))) == ("A", "B"))
    assert ab.certification_status == "descriptive", \
        "< 5 spatial blocks is an invalid spatial null → must not certify a discovery as selected"


def test_true_iid_fallback_when_no_spatial_or_temporal_structure():
    """A single spatial block AND a single time slice (a degenerate cross-section) has no
    block structure on either axis, so the scan honestly falls back to the iid permutation."""
    field = _clustered_independent_field(single_block=True)
    # collapse to one time slice → n_temporal_blocks == 1 as well
    field = GaussianField(
        variables=field.variables, space_ids=field.space_ids, time_ids=(0,),
        Z=field.Z[:, :, :1], W=field.W[:, :, :1], resolution="year",
    )
    records = scan_residual_nonlinear_edges(field, precision=np.eye(3), permutations=300, seed=0)
    assert all(r.null_strategy == "iid_permutation" for r in records)


def test_multiresolution_coarsening_delivers_the_layer_instead_of_refusing(monkeypatch):
    """§II.7/§III.6: when the fine-cell HSIC cache would exceed the memory budget, the scan
    COARSENS to (spatial-block × time-bucket) groups and runs there — the nonlinear layer is
    delivered at a coarser grain (marked in the record warnings), not silently skipped. The
    coarse pass detects COARSE-grain structure, so a group-level nonlinear dependence (each
    UF×year group has a latent g with mean_A≈g, mean_B≈g²) is planted and must survive the
    aggregation (fine-grain structure is left for fine refinement, per §II.7)."""
    import pegasus.ldo.residual_scan as rs

    rng = np.random.default_rng(3)
    n_uf, per_uf, T = 8, 20, 8
    S = n_uf * per_uf
    uf_of_muni = np.repeat(np.arange(n_uf), per_uf)
    Z = np.empty((3, S, T))
    for u in range(n_uf):
        munis = np.where(uf_of_muni == u)[0]
        for t in range(T):
            g = rng.standard_normal()  # per (UF, year) group latent
            Z[0, munis, t] = g + 0.15 * rng.standard_normal(len(munis))       # mean_A ≈ g
            Z[1, munis, t] = g**2 + 0.15 * rng.standard_normal(len(munis))    # mean_B ≈ g² (nonlinear)
            Z[2, munis, t] = rng.standard_normal(len(munis))
    field = GaussianField(
        variables=("A", "B", "N"),
        space_ids=tuple(f"{27 + uf_of_muni[s]}{s:05d}"[:7] for s in range(S)),
        time_ids=tuple(range(T)), Z=Z, W=np.ones_like(Z), resolution="year",
    )
    # Force the fine scan to exceed the budget so the coarsening branch runs, but leave the
    # coarse (n = n_uf × T ≈ 64 groups) scan comfortably under it.
    monkeypatch.setattr(rs, "_residual_scan_memory_budget", lambda: 40_000_000)  # 40 MB

    records = scan_residual_nonlinear_edges(field, np.eye(3), permutations=200, seed=0)
    # the layer RAN (did not refuse) and every emitted edge is tagged as coarsened
    assert records, "coarsened scan returned no edges — the planted A-B dependence was lost"
    assert all(any(w.startswith("multiresolution_coarsened") for w in r.warnings) for r in records)
    assert ("A", "B") in _all_pairs(records)
