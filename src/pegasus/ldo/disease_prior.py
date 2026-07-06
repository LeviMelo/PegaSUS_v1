"""Disease-structure prior on the LDO precision (MSD-III §II.6/§III.4/§5.2; MII-DIS-04).

MSD-III §III.4(5) names three prior-regularizers that "shrink estimates across
neighbours/parents": the spatial GMRF ``L_W``, the disease Laplacian ``L_D``, and
temporal smoothness. This module realizes ``L_D`` in its **adaptive-ℓ1 edge-selection
form**: structurally related disease-concept variables (siblings / parent-child in the
CID-10 hierarchy) get a *lower* ℓ1 penalty on the sparse component ``S`` of the
LVGLASSO, so their sparse edges face a lower selection threshold and survive.

Scope, stated precisely (do not overclaim): this lowers the *selection threshold* for
related-disease edges — it does **not** shrink the dependency *profiles* of related
diseases toward each other. The full §III.4(5) Laplacian-*quadratic* form
``+(γ/2)·tr(Sᵀ L_D S)`` (which would smooth connection profiles across the hierarchy,
the direct analogue of ``L_W``'s spatial whitening in ``lags.fit_lagged_links``) is a
scoped enhancement, **DIS-04b**, not yet implemented — adding it means a smoothness
prox in the ADMM S-step and must not perturb the CPW S/L split at ``γ=0``.

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
    # Gather the variables that map to a graph node; pull their pairwise affinity block
    # in one ``np.ix_`` slice instead of an O(p²) Python double-loop over variable pairs.
    mapped = [(a, code_pos[variables[a]]) for a in range(p) if variables[a] in code_pos]
    if not mapped:
        return affinity
    var_idx = np.fromiter((a for a, _ in mapped), dtype=np.int64, count=len(mapped))
    node_idx = np.fromiter((n for _, n in mapped), dtype=np.int64, count=len(mapped))
    block = np.minimum(dense[np.ix_(node_idx, node_idx)], 1.0)  # affinity capped at 1
    np.fill_diagonal(block, 0.0)                                 # no self-affinity (a<b only before)
    affinity[np.ix_(var_idx, var_idx)] = block
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
