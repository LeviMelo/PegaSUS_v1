"""DIS-04 — the disease L_D prior on the LDO precision (MSD-III §II.6/§III.4/§5.2).

The disease hierarchy enters ``Ω_var`` as a disease-informed adaptive ℓ1 penalty:
structurally related disease-concept variables get a lower penalty so their sparse
links survive. Demonstrated on ground truth — the prior selectively strengthens a
related-disease edge over an equally-strong *unrelated* edge, and fabricates nothing.
"""

from __future__ import annotations

import numpy as np

from pegasus.pirs.ldo.assemble import LDOField
from pegasus.pirs.ldo.disease_prior import disease_penalty_matrix, penalty_from_affinity
from pegasus.pirs.ldo.lowrank import fit_sparse_plus_lowrank
from pegasus.pirs.ldo.orchestrator import run_ldo


def _partial(fit) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(fit.S), 1e-12, None))
    return -fit.S / np.outer(d, d)


def test_disease_prior_selectively_strengthens_related_edge() -> None:
    rng = np.random.default_rng(0)
    p = 6  # A-B related edge; C-D unrelated edge of EQUAL strength; E,F noise
    cov = np.eye(p)
    cov[0, 1] = cov[1, 0] = 0.4
    cov[2, 3] = cov[3, 2] = 0.4
    corr = np.corrcoef(rng.multivariate_normal(np.zeros(p), cov, size=500), rowvar=False)

    lam = 0.18
    affinity = np.zeros((p, p))
    affinity[0, 1] = affinity[1, 0] = 1.0  # only A-B are disease-related
    penalty = penalty_from_affinity(affinity, lambda1=lam)

    no_prior = _partial(fit_sparse_plus_lowrank(corr, lambda1=lam, lambda2=0.5))
    with_prior = _partial(fit_sparse_plus_lowrank(corr, lambda1=lam, lambda2=0.5, penalty_matrix=penalty))

    ab_no, cd_no = abs(no_prior[0, 1]), abs(no_prior[2, 3])
    ab_yes, cd_yes = abs(with_prior[0, 1]), abs(with_prior[2, 3])

    # Without the prior the two equal-strength edges are treated symmetrically.
    assert abs(ab_no - cd_no) < 0.1
    # The prior strengthens the related edge...
    assert ab_yes > ab_no
    # ...and leaves the unrelated edge essentially untouched (no fabricated coupling).
    assert abs(cd_yes - cd_no) < 0.05
    # ...so the related edge now dominates the equally-true unrelated one.
    assert ab_yes > cd_yes + 0.1


def test_penalty_matrix_from_real_cid10_hierarchy() -> None:
    from pegasus.disease.graph import DiseaseGraph

    # I21 & I22 share ICD block I20-I25 (ischaemic heart); A90 (dengue) is another chapter.
    codes = ("I21", "I22", "A90")
    graph = DiseaseGraph.hierarchy(codes)
    lam = 0.2
    penalty = disease_penalty_matrix(codes, graph, lambda1=lam)

    assert penalty[0, 1] < lam                 # related block → discounted penalty
    assert penalty[0, 2] == lam                # cross-chapter → full penalty
    assert penalty[1, 2] == lam


def test_run_ldo_threads_disease_graph() -> None:
    from pegasus.disease.graph import DiseaseGraph

    codes = ("I21", "I22", "A90")
    rng = np.random.default_rng(2)
    S, T = 16, 30
    X = rng.standard_normal((3, S, T))
    field = LDOField(
        variables=codes,
        space_ids=tuple(str(270000 + i) for i in range(S)),
        time_ids=tuple(range(T)),
        X=X,
        W=np.ones_like(X),
        resolution="month",
    )
    graph = DiseaseGraph.hierarchy(codes)
    run = run_ldo(field, K=2, n_subsamples=4, run_residual_scan=False, seed=0, disease_graph=graph)
    # the wiring flows end-to-end: a valid run with the same variable set
    assert run.variables == codes
    assert isinstance(run.link_records, list)
