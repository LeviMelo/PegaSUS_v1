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


def disease_laplacian_matrix(variables: tuple[str, ...], graph: "DiseaseGraph") -> np.ndarray:
    """``p×p`` disease-graph Laplacian ``L_D = D − W`` aligned to the LDO variable set.

    ``W`` is the structural affinity (:func:`variable_affinity`: 1 same category, 0.5 same
    block, 0.25 same chapter; 0 for unmatched variables). ``L_D`` is the §III.4(5) disease-
    smoothness operator: the quadratic ``+(γ/2)·tr(Sᵀ L_D S)`` shrinks each variable's
    precision row toward those of its hierarchical neighbours (structurally-close diseases
    get similar dependency profiles). Variables absent from the graph are isolated rows of
    ``L_D`` (zero), so they receive no disease smoothing. PSD by construction.
    """
    W = variable_affinity(variables, graph)  # symmetric, zero diagonal, ≥0
    return np.diag(W.sum(axis=1)) - W


def _laplacian_from_mask(mask: np.ndarray) -> np.ndarray:
    A = mask.astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return np.diag(A.sum(axis=1)) - A


def hierarchical_disease_laplacians_from_affinity(affinity: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose the graded structural affinity into PER-SCALE Laplacians (§III.3 sum-of-scales).

    The graded affinity carries three nested scales at distinct weights (1 same category, 0.5 same
    block, 0.25 same chapter). This splits them into separate same-scale adjacency Laplacians
    ``{category, block, chapter}`` so each scale can carry its OWN precision — the sum-of-scales
    GMRF ``θ_leaf = μ_chapter + δ_block + δ_category + δ_leaf`` with distinct ``τ_level`` shrinkage,
    rather than one flat Laplacian at a single scalar precision. Empty scales are omitted."""
    W = np.asarray(affinity, dtype=np.float64)
    bands = {
        "category": W >= 0.99,
        "block": (W >= 0.4) & (W < 0.99),
        "chapter": (W >= 0.15) & (W < 0.4),
    }
    out: dict[str, np.ndarray] = {}
    for name, mask in bands.items():
        if mask.any():
            out[name] = _laplacian_from_mask(mask)
    return out


def hierarchical_disease_laplacians(variables: tuple[str, ...], graph: "DiseaseGraph") -> dict[str, np.ndarray]:
    """Per-scale disease Laplacians aligned to the LDO variables (see
    :func:`hierarchical_disease_laplacians_from_affinity`)."""
    return hierarchical_disease_laplacians_from_affinity(variable_affinity(variables, graph))


def sum_of_scales_disease_operator(
    variables: tuple[str, ...], graph: "DiseaseGraph",
    scale_precisions: dict[str, float] | None = None, *,
    adaptive: bool = False,
    field_values_by_variable: dict[str, np.ndarray] | None = None,
) -> np.ndarray | None:
    """The §III.3 sum-of-scales disease GMRF operator ``G_D = Σ_level τ_level · L_level``.

    Distinct per-scale precisions (e.g. a rare leaf shrinks strongly toward its category but weakly
    toward its chapter) replace the flat single-γ Laplacian. Returns ``None`` when no scale has a
    positive precision or the graph induces no coupling — the caller then falls back to the flat
    ``L_D`` (or no disease prior).

    Each ``L_level`` is a graph Laplacian (zero row sums), so ``G_D`` penalises only *differences*
    between related-disease precision rows — cross-disease smoothness, never a variable's own level
    (``xᵀG_D x`` is invariant to a constant shift of ``x``). Requirement (1) holds by construction.

    ``adaptive=True`` (opt-in): estimate the per-scale precisions from ``field_values_by_variable``
    by empirical Bayes (:func:`estimate_disease_scale_precisions`) instead of the fixed
    ``scale_precisions`` constant, so genuinely-divergent blocks override the prior. The default
    (``adaptive=False`` with explicit ``scale_precisions``) reproduces the original behavior exactly.
    """
    laps = hierarchical_disease_laplacians(variables, graph)
    if not laps:
        return None
    if adaptive:
        if field_values_by_variable is None:
            raise ValueError("adaptive=True requires field_values_by_variable")
        precisions = estimate_disease_scale_precisions(field_values_by_variable, variables, graph)
    else:
        precisions = scale_precisions or {}
    p = len(variables)
    G = np.zeros((p, p), dtype=np.float64)
    used = False
    for name, L in laps.items():
        tau = float(precisions.get(name, 0.0))
        if tau > 0.0:
            G = G + tau * L
            used = True
    return G if used else None


def _scale_masks(variables: tuple[str, ...], graph: "DiseaseGraph") -> dict[str, np.ndarray]:
    """Per-scale boolean adjacency masks (same-category / same-block / same-chapter), aligned to
    the LDO variables. A block of related diseases at a scale is a connected component of its mask."""
    W = variable_affinity(variables, graph)
    return {
        "category": W >= 0.99,
        "block": (W >= 0.4) & (W < 0.99),
        "chapter": (W >= 0.15) & (W < 0.4),
    }


def _connected_components(mask: np.ndarray) -> list[list[int]]:
    """Blocks of mutually-related variables: connected components of a symmetric adjacency mask
    (isolated singletons omitted — a block needs ≥2 members to borrow strength)."""
    A = np.asarray(mask, dtype=bool).copy()
    np.fill_diagonal(A, False)
    n = A.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start] or not A[start].any():
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(A[u] & ~seen)[0]:
                seen[v] = True
                stack.append(int(v))
        if len(comp) >= 2:
            comps.append(sorted(comp))
    return comps


def _block_tau2(values: list[np.ndarray]) -> float:
    """Method-of-moments empirical-Bayes shrinkage precision for one block of related diseases.

    ``τ² = max(0, between-disease var of the block's per-disease means − mean within-disease noise)``
    normalized to the within-noise scale. A block whose diseases are genuinely alike (between ≈ noise)
    → large τ² (strong shrinkage: the prior wins). A block whose diseases genuinely differ (between ≫
    noise) → τ² → 0 (weak shrinkage: the data overrides the prior)."""
    means = np.array([float(np.nanmean(v)) for v in values if v.size], dtype=np.float64)
    if means.size < 2:
        return 0.0
    within = np.array(
        [float(np.nanvar(v, ddof=1)) / max(1, v.size) for v in values if v.size > 1],
        dtype=np.float64,
    )
    noise = float(np.mean(within)) if within.size else 0.0
    between = float(np.var(means, ddof=1))
    signal = max(0.0, between - noise)              # true between-disease dispersion
    return noise / (noise + signal) if (noise + signal) > 0 else 0.0


def estimate_disease_scale_precisions(
    field_values_by_variable: dict[str, np.ndarray],
    variables: tuple[str, ...],
    graph: "DiseaseGraph",
) -> dict[str, float]:
    """Empirical-Bayes per-scale shrinkage precisions τ²_level estimated FROM THE DATA (§III.3).

    For each scale (category/block/chapter) and each block of mutually-related variables at that
    scale, method-of-moments ``τ² = noise / (noise + max(0, between − noise))`` (:func:`_block_tau2`):
    ≈1 when the related diseases genuinely agree (shrink hard), →0 when they genuinely diverge (let
    the data speak). The scale's precision is the mean block τ² (scales/blocks with no data or no
    ≥2-member block contribute nothing)."""
    masks = _scale_masks(variables, graph)
    vals = [np.asarray(field_values_by_variable.get(v, np.empty(0)), float).ravel() for v in variables]
    out: dict[str, float] = {}
    for name, mask in masks.items():
        taus = [_block_tau2([vals[i] for i in comp]) for comp in _connected_components(mask)]
        if taus:
            out[name] = float(np.mean(taus))
    return out


def heavily_shrunk_blocks(
    field_values_by_variable: dict[str, np.ndarray],
    variables: tuple[str, ...],
    graph: "DiseaseGraph",
    *, threshold: float = 0.75,
) -> list[dict]:
    """Flag blocks whose adaptive τ² exceeds ``threshold`` — a downstream consumer warns that these
    variables' estimates are heavily borrowed from their hierarchical neighbours (§III.4(5)). Each
    entry: ``{scale, variables, tau2}``."""
    masks = _scale_masks(variables, graph)
    vals = [np.asarray(field_values_by_variable.get(v, np.empty(0)), float).ravel() for v in variables]
    flagged: list[dict] = []
    for name, mask in masks.items():
        for comp in _connected_components(mask):
            tau2 = _block_tau2([vals[i] for i in comp])
            if tau2 > threshold:
                flagged.append({
                    "scale": name,
                    "variables": tuple(variables[i] for i in comp),
                    "tau2": tau2,
                })
    return flagged


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
    "disease_laplacian_matrix",
    "hierarchical_disease_laplacians_from_affinity",
    "hierarchical_disease_laplacians",
    "sum_of_scales_disease_operator",
    "estimate_disease_scale_precisions",
    "heavily_shrunk_blocks",
    "penalty_from_affinity",
    "disease_penalty_matrix",
    "tile_penalty_across_lags",
]
