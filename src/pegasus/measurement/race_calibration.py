"""Ecological calibration of the RaceBridge prior from real data (MSD §4.6, W-RACE-2 wiring).

The race-bridge registry's default prior is an IDENTITY emission (administrative race taken as
self-declared, no reclassification correction) — honest but uncalibrated, and its own provenance note
asks to "swap in a linkage-derived emission matrix for reclassification-corrected estimates". Record
linkage does not exist and individual reclassification is unidentifiable, so the identifiable path is
**ecological**: pool administrative-race event counts against the census self-declared population
across cells whose composition varies, estimate the confusion matrix, and write it back as a
calibrated prior the existing local-pi bridge consumes unchanged.

This module is the wire. It does NOT restructure the live per-cell bridge or the population tensor
(the population race composition is census-anchored and dominant; the bridged flows are a second-order
correction). It replaces the *placeholder C* — the actual defect the user named ("the degenerate
plug-in") — with a data-estimated one, and refuses to emit when the data cannot identify it
(CLAUDE.md IV/V: never silently degrade). The estimator it calls is validated by planted-signal
recovery (test_race_ecological_deconvolution.py).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pegasus.measurement.race import validate_race_bridge_prior, RaceBridgePrior
from pegasus.measurement.race_ecological import (
    EcologicalRaceProblem,
    IDENTIFIABILITY_FLOOR,
    contextual_identifiability,
    fit_ecological_race_deconvolution,
    reclassification_from_emission,
)


class RaceCalibrationError(ValueError):
    """Raised when the data cannot support an ecological race-bridge calibration."""


def calibrate_race_bridge_prior(
    *,
    admin_counts_by_cell: np.ndarray,
    census_pop_by_cell: np.ndarray,
    admin_codes: list[str],
    target_categories: list[str],
    bridge_id: str,
    source_system: str = "SIM-DO",
    source_axis: str = "SIM_ADMIN_RACACOR",
    target_axis: str = "IBGE_SELF_DECLARED_RACE",
    calibration_scope: str = "ecological_national",
    prior_strength: float = 2.0,
    min_identifiability: float = IDENTIFIABILITY_FLOOR,
    bootstrap_replicates: int = 0,
) -> dict[str, Any]:
    """Estimate the admin→self-declared confusion ecologically and return a calibrated Bridge_R prior.

    Args:
        admin_counts_by_cell: (S, K) administrative-race event counts per cell (e.g. SIM deaths per
            municipality×year by RACACOR code).
        census_pop_by_cell: (S, J) census self-declared population per cell (the exposure/denominator).
        admin_codes: length-K administrative codes, in the column order of ``admin_counts_by_cell``
            (the registry's ``source_categories``, e.g. ``["1","2","3","4","5"]``).
        target_categories: length-J self-declared category names, in the column order of
            ``census_pop_by_cell`` (the registry's ``target_categories``).
        bridge_id: id for the emitted prior (must match the registry entry that will reference it).
        calibration_scope: free-text scope tag recorded in metadata (e.g. ``ecological_AL_2018_2022``).
        prior_strength: kappa toward an identity emission prior; keep weak (a large value re-erases
            the signal — see the estimator's validation).
        min_identifiability: refuse to emit below this contextual-identifiability floor rather than
            ship a prior-dominated (near-identity) calibration masquerading as data-calibrated.
        bootstrap_replicates: if >0, a parametric bootstrap CV is folded into ``sensitivity_width``.

    Returns:
        A Bridge_R prior payload (already passed through :func:`validate_race_bridge_prior`) with the
        row-stochastic reclassification matrix ``P(self|admin)`` and ecological provenance metadata.

    Raises:
        RaceCalibrationError: shapes inconsistent, or the cell compositions are too collinear to
            identify the confusion (``contextual_identifiability < min_identifiability``).
    """
    Y = np.asarray(admin_counts_by_cell, dtype=float)
    N = np.asarray(census_pop_by_cell, dtype=float)
    K = len(admin_codes)
    J = len(target_categories)
    if Y.ndim != 2 or N.ndim != 2:
        raise RaceCalibrationError("admin_counts_by_cell and census_pop_by_cell must be 2-D.")
    if Y.shape[1] != K:
        raise RaceCalibrationError(f"admin_counts has {Y.shape[1]} cols but {K} admin_codes.")
    if N.shape[1] != J:
        raise RaceCalibrationError(f"census_pop has {N.shape[1]} cols but {J} target_categories.")
    if Y.shape[0] != N.shape[0]:
        raise RaceCalibrationError(f"cell count mismatch: admin {Y.shape[0]} vs census {N.shape[0]}.")

    ident = contextual_identifiability(N)
    if ident < min_identifiability:
        raise RaceCalibrationError(
            f"cell compositions too collinear to identify the confusion "
            f"(contextual_identifiability={ident:.4f} < {min_identifiability}); "
            "a calibration here would be prior-dominated (near-identity) and dishonest. "
            "Widen the cell set (more municipalities / more compositional variation) or keep the "
            "honest identity baseline."
        )

    problem = EcologicalRaceProblem(
        N=N,
        Y_by_stratum=[Y],
        self_declared_categories=list(target_categories),
        admin_categories=list(admin_codes),
        stratum_labels=[source_system],
    )
    result = fit_ecological_race_deconvolution(
        problem, prior_strength=prior_strength, bootstrap_replicates=bootstrap_replicates
    )

    C = result.emission_by_stratum[source_system]  # (K, J) P(admin=k | self=j), column-stochastic
    p_self = N.sum(axis=0)
    if p_self.sum() <= 0:
        raise RaceCalibrationError("census population has zero total mass.")
    R = reclassification_from_emission(C, p_self)  # (K, J) P(self=j | admin=k), row-stochastic

    # sensitivity_width carries the calibration's own uncertainty: at least the identity-gap the
    # calibration moved away from (so a barely-moved C stays honest about being near-identity), and,
    # if bootstrapped, the worst rate-ratio CV.
    identity_gap = float(np.abs(R - _identity_reclassification(K, J)).max())
    width = min(1.0, max(0.05, identity_gap))
    if result.rate_ratio_cv:
        width = min(1.0, max(width, max(result.rate_ratio_cv.values())))

    matrix = {
        admin_codes[k]: {target_categories[j]: float(R[k, j]) for j in range(J)}
        for k in range(K)
    }
    payload: dict[str, Any] = {
        "bridge_id": bridge_id,
        "mode": "localPi_posteriorC",
        "source_axis": source_axis,
        "target_axis": target_axis,
        "source_categories": list(admin_codes),
        "target_categories": list(target_categories),
        "sensitivity_width": round(width, 4),
        "metadata": {
            "epistemic_status": "ecologically_calibrated",
            "calibration_scope": calibration_scope,
            "calibration_method": "hierarchical_poisson_ecological_deconvolution",
            "contextual_identifiability": round(float(ident), 4),
            "n_cells": int(N.shape[0]),
            "prior_strength": float(prior_strength),
            "matrix_concentration": 250.0,
            "local_pi_concentration": 250.0,
            "provenance": (
                "Ecological deconvolution (MSD §4.6): confusion estimated by pooling administrative "
                "race event counts against census self-declared population across cells with varying "
                "composition (Goodman/King ecological identifiability), then written back as "
                "P(self|admin) via Bayes with the census self marginal. NOT record-linked; a "
                "population-level reclassification correction, not an individual one."
            ),
            "warning": (
                "Ecologically-calibrated reclassification prior. Posterior race counts remain "
                "bridge-derived observer fields, not direct self-declared measurements; raw "
                "administrative counts must be preserved."
            ),
        },
        "matrix": matrix,
    }
    # Fail loudly here rather than at load time if we ever emit an illegal matrix.
    validate_race_bridge_prior(payload)
    return payload


def _identity_reclassification(K: int, J: int) -> np.ndarray:
    R = np.zeros((K, J), dtype=float)
    for d in range(min(K, J)):
        R[d, d] = 1.0
    # any admin code with no diagonal target gets a uniform row so the comparison is well-defined
    zero_rows = R.sum(axis=1) <= 0
    if np.any(zero_rows):
        R[zero_rows, :] = 1.0 / J
    return R


def emission_of(prior: RaceBridgePrior, admin_marginal: np.ndarray) -> np.ndarray:
    """The emission ``P(admin|self)`` implied by a stored (reclassification) prior + an admin marginal.

    Convenience for inspecting/round-tripping a calibrated prior: reads the registry matrix in its
    stored ``matrix[admin][self]`` orientation and converts back to the emission direction.
    """
    from pegasus.measurement.race_ecological import emission_from_reclassification

    K = len(prior.source_categories)
    J = len(prior.target_categories)
    R = np.zeros((K, J), dtype=float)
    for k, src in enumerate(prior.source_categories):
        row = prior.matrix[src]
        for j, tgt in enumerate(prior.target_categories):
            R[k, j] = float(row.get(tgt, 0.0))
    return emission_from_reclassification(R, np.asarray(admin_marginal, dtype=float))


__all__ = ["calibrate_race_bridge_prior", "emission_of", "RaceCalibrationError"]
