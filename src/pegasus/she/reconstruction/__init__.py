"""Constrained Tensor Reconstruction (CTR) kernel — MSD-II §II.3."""

from pegasus.she.reconstruction.certify import (
    CTRCertificationError,
    CTRCertificationPolicy,
    CTRCertificationRow,
    assert_ctr_verified_promotion_allowed,
    certify_ctr,
)
from pegasus.she.reconstruction.instances import (
    age_bin_disaggregation_instance,
    population_ctr_instance,
)
from pegasus.she.reconstruction.problem import (
    CTREvaluation,
    CTRProblem,
    MarginalConstraint,
    ObservationTerm,
    QuadraticPenalty,
    evaluate_ctr,
    solve_ctr,
)

__all__ = [
    "CTRProblem",
    "ObservationTerm",
    "QuadraticPenalty",
    "MarginalConstraint",
    "CTREvaluation",
    "evaluate_ctr",
    "solve_ctr",
    "population_ctr_instance",
    "age_bin_disaggregation_instance",
    "certify_ctr",
    "assert_ctr_verified_promotion_allowed",
    "CTRCertificationPolicy",
    "CTRCertificationRow",
    "CTRCertificationError",
]
