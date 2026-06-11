from __future__ import annotations

from typing import Any


def stdfm_objective_pseudocode_contract() -> dict[str, Any]:
    """Typed formula-to-code contract for the gated ST-DFM objective.

    This is not a numerical solver. It is the required disambiguated bridge
    between the MSD formula and future production PyTorch implementation.
    """
    return {
        "inputs": {
            "Y": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64", "missing": "mask"},
            "M": {"shape": "[n_space, n_time, n_fields]", "dtype": "bool", "meaning": "observed mask"},
            "Z": {"shape": "[n_space, n_time, n_covariates]", "dtype": "float64"},
            "support": "municipality/year support aligned before invocation",
        },
        "outputs": {
            "latent_factor": {"shape": "[n_space, n_time, k]", "dtype": "float64"},
            "reconstruction": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64"},
            "uncertainty": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64"},
        },
        "epsilon_stabilization": "variance and bounded-link denominators clamp at eps=1e-9",
        "warnings": ["stdfm_certification_required", "blocked_solver_pending"],
        "failure_modes": [
            "insufficient_temporal_points",
            "concept_incompatibility",
            "uncertified_verified_promotion",
        ],
    }
