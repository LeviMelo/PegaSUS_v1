"""Disease-structure prior on the LDO precision (MSD-III §II.6/§III.4/§5.2; MII-DIS-04).

The disease Laplacian ``L_D`` enters the variable-dependency operator ``Ω_var`` as a
smoothness/fused prior — the disease-axis analogue of the spatial GMRF ``L_W`` on the
cell precision. It is realized as a disease-informed *adaptive ℓ1 penalty* on the
sparse component ``S`` of the LVGLASSO: structurally related disease-concept variables
(siblings / parent-child in the CID-10 hierarchy) get a *lower* penalty, so their
(sparse) edges survive — "dependency profiles vary smoothly across semantically
similar diseases" (§5.2).

Only ``structural`` DiseaseGraphs are admissible as a prior; a ``context_derived``
co-occurrence graph is refused by ``DiseaseGraph.as_prior`` (the circularity guard).
When no graph is supplied the estimator falls back to the scalar penalty — an honest
no-op, never a fabricated coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # keep the LDO import-light — the disease/ICD adapter is an optional extra
    from pegasus.disease.graph import DiseaseGraph


def variable_affinity(variables: tuple[str, ...], graph: "DiseaseGraph") -> np.ndarray:
    """``p×p`` structural affinity in [0,1] between LDO variables.

    A variable id is matched to a disease code by name; the affinity is the graph's
    structural edge weight (1 same category, 0.5 same block, 0.25 same chapter).
    Variables absent from the graph contribute zero affinity (no prior coupling).
    """
    p = len(variables)
    affinity = np.zeros((p, p), dtype=np.float64)
    if graph.legality_class != "structural":
        # Guard mirrored from DiseaseGraph.as_prior: a context-derived graph must not
        # inform an edge on the same data. Refuse loudly rather than leak a prior.
        graph.as_prior()  # raises DiseaseGraphCircularityError
    code_pos = {c: i for i, c in enumerate(graph.codes)}
    dense = graph.adjacency().toarray()
    node_of = {v: code_pos[v] for v in variables if v in code_pos}
    for a in range(p):
        na = node_of.get(variables[a])
        if na is None:
            continue
        for b in range(a + 1, p):
            nb = node_of.get(variables[b])
            if nb is None:
                continue
            w = float(dense[na, nb])
            if w > 0.0:
                affinity[a, b] = affinity[b, a] = min(1.0, w)
    return affinity


def penalty_from_affinity(
    affinity: np.ndarray, *, lambda1: float, beta: float = 0.7, min_frac: float = 0.2
) -> np.ndarray:
    """Per-pair ℓ1 penalty ``λ1·(1 − β·affinity)`` floored at ``λ1·min_frac``.

    ``affinity=1`` (same category) → penalty ``λ1·(1−β)``; ``affinity=0`` → ``λ1``.
    The floor keeps every penalty strictly positive so the estimate stays sparse/PD.
    """
    affinity = np.clip(np.asarray(affinity, dtype=np.float64), 0.0, 1.0)
    penalty = lambda1 * (1.0 - beta * affinity)
    return np.clip(penalty, lambda1 * float(min_frac), lambda1)


def disease_penalty_matrix(
    variables: tuple[str, ...], graph: "DiseaseGraph", *, lambda1: float,
    beta: float = 0.7, min_frac: float = 0.2,
) -> np.ndarray:
    """The ``p×p`` disease-informed penalty over the LDO variable set."""
    return penalty_from_affinity(
        variable_affinity(variables, graph), lambda1=lambda1, beta=beta, min_frac=min_frac
    )


def tile_penalty_across_lags(penalty_pp: np.ndarray, K: int, lambda1: float) -> np.ndarray:
    """Expand a ``p×p`` variable penalty to ``(p·(K+1))²`` for the lag-extended features.

    The lag feature index is ``f = lag·p + var`` (see ``lags._build_lagged_feature_matrix``);
    disease relatedness is lag-invariant, so feature ``(lag a, var i)`` vs ``(lag b, var j)``
    inherits ``penalty_pp[i, j]`` in every (within- and cross-lag) block.
    """
    p = penalty_pp.shape[0]
    big = np.full((p * (K + 1), p * (K + 1)), float(lambda1), dtype=np.float64)
    for a in range(K + 1):
        for b in range(K + 1):
            big[a * p:(a + 1) * p, b * p:(b + 1) * p] = penalty_pp
    return big


__all__ = [
    "variable_affinity",
    "penalty_from_affinity",
    "disease_penalty_matrix",
    "tile_penalty_across_lags",
]
