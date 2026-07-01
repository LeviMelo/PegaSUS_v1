"""CTR instances (MSD-II §II.3, MII-CTR-01/02).

- ``population_ctr_instance``: the canonical instance. Its latent is ``[P | M]``
  (population then migration, each of length ``n_cells``); its ``native_evaluator``
  delegates to ``she.population.loss.evaluate_population_loss`` byte-for-byte, so
  the CTR kernel reproduces the population tensor exactly (golden-value equal).
- ``age_bin_disaggregation_instance``: a genuinely-linear instance (MII-CTR-02).
  Recover fine age-bin counts ``n_fine`` from broad-bin totals under the
  aggregation constraint ``A n_fine = n_broad`` plus 2nd-difference smoothness
  (and an optional cohort/level anchor). Uses only the generic CTR core.
"""

from __future__ import annotations

import numpy as np

from pegasus.she.population.loss import evaluate_population_loss
from pegasus.she.population.schema import PopulationTensorProblem
from pegasus.she.reconstruction.problem import (
    CTRProblem,
    MarginalConstraint,
    ObservationTerm,
    QuadraticPenalty,
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


__all__ = ["population_ctr_instance", "age_bin_disaggregation_instance"]
