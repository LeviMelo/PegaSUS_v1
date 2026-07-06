"""CTR instances (MSD-II §II.3, MII-CTR-01/02).

- ``population_ctr_instance``: the canonical instance. Its latent is ``[P | M]``
  (population then migration, each of length ``n_cells``); its ``native_evaluator``
  delegates to ``she.reconstruction.loss.evaluate_population_loss`` byte-for-byte, so
  the CTR kernel reproduces the population tensor exactly (golden-value equal).
- ``age_bin_disaggregation_instance``: a genuinely-linear instance (MII-CTR-02).
  Recover fine age-bin counts ``n_fine`` from broad-bin totals under the
  aggregation constraint ``A n_fine = n_broad`` plus 2nd-difference smoothness
  (and an optional cohort/level anchor). Uses only the generic CTR core.
- ``migration_flow_instance``: reconstruct a directed origin→destination migration
  flow vector over candidate pairs from (a) an observed per-node NET-migration
  marginal, (b) a gravity structural prior, and (c) an optional census O→D anchor.
  A genuinely-linear instance (all terms quadratic) on the generic core. See
  ``sidra.population_cube.migration`` for the data-facing builder.
"""

from __future__ import annotations

import numpy as np

from pegasus.denominators.reconstruction.loss import evaluate_population_loss
from pegasus.denominators.reconstruction.schema import PopulationTensorProblem
from pegasus.denominators.reconstruction.problem import (
    CTRProblem,
    MarginalConstraint,
    ObservationTerm,
    QuadraticPenalty,
)


def net_flow_operator(n_nodes: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """The (n_nodes × P) net operator S: ``(S F)_j = inflow_j - outflow_j``.

    For directed candidate pair ``p = (i→j)`` (i origin, j destination), column p
    contributes ``+1`` to row j (an arrival) and ``-1`` to row i (a departure).
    Then ``S F`` is the per-node net migration implied by the flow vector F, and
    ``S F = net`` is exactly the demographic balancing identity (MSD §2.8.7).
    """
    S = np.zeros((n_nodes, len(pairs)), dtype=np.float64)
    for p, (i, j) in enumerate(pairs):
        S[j, p] += 1.0
        S[i, p] -= 1.0
    return S


def migration_flow_instance(
    *,
    pairs: list[tuple[int, int]],
    gravity_prior: np.ndarray | list[float],
    net_by_node: np.ndarray | list[float],
    net_weight: float = 10.0,
    gravity_weight: float = 1.0,
    census_anchor_values: np.ndarray | list[float] | None = None,
    census_anchor_mask: np.ndarray | list[bool] | None = None,
    census_weight: float = 50.0,
) -> CTRProblem:
    """O→D migration flow reconstruction as a CTR instance (MSD §2.8.7 flow layer).

    Latent: the non-negative flow ``F`` over the ``P`` directed candidate ``pairs``.
    Terms:
      * gravity prior   ``gravity_weight * ||F - G||^2`` — structural prior that
        resolves the (severe) null space of the net-only marginal; G is the
        gravity expectation per pair.
      * net marginal    ``net_weight * ||S F - net||^2`` — the balancing identity,
        the only *observed* constraint when no census O→D table is available.
      * census anchor   ``census_weight * ||F - Fcensus||^2`` over anchored pairs —
        the genuine bilateral observation at census years (optional).

    Without a census anchor this is a gravity-structured, net-consistent *estimate*
    (the net marginal fixes each node's level, gravity fixes the relative
    allocation); with it, an interpolation pinned to real bilateral flows.
    """
    P = len(pairs)
    gravity = np.asarray(gravity_prior, dtype=np.float64).reshape(-1)
    net = np.asarray(net_by_node, dtype=np.float64).reshape(-1)
    if gravity.size != P:
        raise ValueError("gravity_prior length must equal number of pairs")
    n_nodes = net.size

    observations = [ObservationTerm(A=np.eye(P), values=gravity, weight=gravity_weight, name="gravity")]
    if census_anchor_values is not None and census_anchor_mask is not None:
        mask = np.asarray(census_anchor_mask, dtype=bool).reshape(-1)
        values = np.asarray(census_anchor_values, dtype=np.float64).reshape(-1)
        if mask.size != P or values.size != P:
            raise ValueError("census anchor arrays must have length P")
        rows = np.where(mask)[0]
        if rows.size:
            selector = np.zeros((rows.size, P), dtype=np.float64)
            for r, col in enumerate(rows):
                selector[r, col] = 1.0
            observations.append(ObservationTerm(A=selector, values=values[rows], weight=census_weight, name="census_anchor"))

    S = net_flow_operator(n_nodes, pairs)
    return CTRProblem(
        latent_shape=(P,),
        observations=tuple(observations),
        marginals=(MarginalConstraint(S=S, totals=net, weight=net_weight, name="net_balance"),),
        nonnegative=True,
        metadata={"instance": "migration_flow", "n_nodes": n_nodes, "n_pairs": P},
    )


def population_ctr_instance(problem: PopulationTensorProblem) -> CTRProblem:
    """Wrap the population objective as a CTR instance (native delegation)."""
    n = problem.n_cells

    def native(x: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
        population = x[:n]
        migration = x[n:]
        ev = evaluate_population_loss(problem, tuple(population), tuple(migration))
        grad = np.concatenate([np.asarray(ev.population_gradient), np.asarray(ev.migration_gradient)])
        return ev.total, grad, dict(ev.terms)

    return CTRProblem(
        latent_shape=(2 * n,),
        native_evaluator=native,
        nonnegative=True,
        metadata={"instance": "population_tensor", "mode": problem.mode, "cells": n},
    )


def _second_difference_matrix(m: int) -> np.ndarray:
    """(m-2) x m 2nd-difference operator; empty if m < 3."""
    if m < 3:
        return np.zeros((0, m), dtype=np.float64)
    D = np.zeros((m - 2, m), dtype=np.float64)
    for i in range(m - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def age_bin_disaggregation_instance(
    broad_totals: np.ndarray | list[float],
    aggregation: np.ndarray,
    *,
    totals_weight: float = 100.0,
    smoothness_weight: float = 1.0,
    level_anchor: np.ndarray | list[float] | None = None,
    level_weight: float = 1e-3,
) -> CTRProblem:
    """Age-bin disaggregation as a CTR instance (MII-CTR-02).

    ``aggregation`` (n_broad x n_fine) sums fine bins into broad bins; the recovered
    ``n_fine`` must satisfy ``aggregation @ n_fine ≈ broad_totals`` (hard-ish via a
    high totals_weight) while being smooth across adjacent fine bins (2nd
    difference). An optional ``level_anchor`` provides a weak prior on the fine
    profile so the null space of the aggregation is resolved.
    """
    aggregation = np.asarray(aggregation, dtype=np.float64)
    broad_totals = np.asarray(broad_totals, dtype=np.float64).reshape(-1)
    n_broad, n_fine = aggregation.shape
    if broad_totals.size != n_broad:
        raise ValueError("broad_totals length must equal aggregation rows")

    observations = [ObservationTerm(A=aggregation, values=broad_totals, weight=totals_weight, name="totals")]
    if level_anchor is not None:
        anchor = np.asarray(level_anchor, dtype=np.float64).reshape(-1)
        if anchor.size != n_fine:
            raise ValueError("level_anchor length must equal n_fine")
        observations.append(ObservationTerm(A=np.eye(n_fine), values=anchor, weight=level_weight, name="level"))

    penalties = ()
    D = _second_difference_matrix(n_fine)
    if D.shape[0] > 0:
        penalties = (QuadraticPenalty(D=D, weight=smoothness_weight, name="age_smooth"),)

    return CTRProblem(
        latent_shape=(n_fine,),
        observations=tuple(observations),
        penalties=penalties,
        marginals=(MarginalConstraint(S=aggregation, totals=broad_totals, weight=0.0, name="closure_check"),),
        nonnegative=True,
        metadata={"instance": "age_bin_disaggregation", "n_broad": n_broad, "n_fine": n_fine},
    )


__all__ = [
    "population_ctr_instance",
    "age_bin_disaggregation_instance",
    "migration_flow_instance",
    "net_flow_operator",
]
