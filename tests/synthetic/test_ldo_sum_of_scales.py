"""O6 / §III.3 — sum-of-scales disease GMRF (per-scale precisions, not one flat γ).

The disease prior must decompose into nested scales — θ_leaf = μ_chapter + δ_block + δ_category +
δ_leaf — each with its OWN precision τ_level, so a rare leaf shrinks strongly toward its category
but weakly toward its chapter. The old L_D was a single flat graded Laplacian at one scalar γ.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.disease_prior import (
    disease_laplacian_matrix,  # noqa: F401  (imported to assert the flat form still exists)
    hierarchical_disease_laplacians_from_affinity,
)


def _affinity():
    # 4 variables: (0,1) same category (1.0); (1,2) same block (0.5); (2,3) same chapter (0.25)
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 0] = 1.0
    W[1, 2] = W[2, 1] = 0.5
    W[2, 3] = W[3, 2] = 0.25
    return W


def test_affinity_splits_into_per_scale_laplacians():
    laps = hierarchical_disease_laplacians_from_affinity(_affinity())
    assert set(laps) == {"category", "block", "chapter"}
    # each scale's Laplacian couples only its own-scale pair
    assert laps["category"][0, 1] == -1.0 and laps["category"][2, 3] == 0.0
    assert laps["block"][1, 2] == -1.0 and laps["block"][0, 1] == 0.0
    assert laps["chapter"][2, 3] == -1.0 and laps["chapter"][1, 2] == 0.0
    for L in laps.values():                     # every scale Laplacian is PSD with zero row sums
        assert np.allclose(L.sum(axis=1), 0.0)
        assert np.linalg.eigvalsh(L).min() > -1e-9


def test_per_scale_precisions_weight_scales_distinctly():
    from pegasus.ldo.disease_prior import _laplacian_from_mask
    laps = hierarchical_disease_laplacians_from_affinity(_affinity())
    # strong category shrinkage, zero chapter shrinkage
    G = 3.0 * laps["category"] + 1.0 * laps["block"] + 0.0 * laps["chapter"]
    # the category pair (0,1) is penalised 3×; the chapter pair (2,3) not at all
    assert G[0, 1] == -3.0
    assert G[2, 3] == 0.0
    # distinct from the flat operator (one weight everywhere a coupling exists)
    flat = laps["category"] + laps["block"] + laps["chapter"]
    assert not np.allclose(G, flat)


def test_run_ldo_accepts_disease_scale_precisions_without_error():
    # smoke: the param threads through run_ldo (no disease graph → falls back cleanly)
    import numpy as np

    from pegasus.ldo.margins import GaussianField
    from pegasus.ldo.orchestrator import run_ldo
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((4, 40, 6))
    Z[1] += 0.7 * Z[0]
    field = GaussianField(variables=("A", "B", "C", "D"),
                          space_ids=tuple(f"27{i:05d}"[:7] for i in range(40)),
                          time_ids=tuple(range(6)), Z=Z, W=np.ones((4, 40, 6)), resolution="year")
    run = run_ldo(field, K=1, n_subsamples=2,
                  disease_scale_precisions={"category": 2.0, "chapter": 0.1}, seed=0)
    assert run.link_records is not None
