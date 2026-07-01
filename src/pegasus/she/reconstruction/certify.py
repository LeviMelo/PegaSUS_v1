"""CTR certification gate (MSD-II §II.3.1, MII-CTR-03).

Generalizes the ST-DFM certification (§2.10.4) to every CTR output. A reconstructed
tensor may enter substrate only as ``B_reconstructed``/``B_latent`` (never
``B_official``), is dashboard-unsafe by default, and may be promoted to
dashboard-safe only when ``verified`` *and* carrying propagated uncertainty.

Thresholds are reused from ``STDFMCertificationPolicy`` (one source of truth) and
extended with a ``constraint_residual`` tolerance (the CTR solve's marginal/
observation feasibility). New §10 abort: "reconstructed tensor promoted to
verified without holdout certification or propagated uncertainty."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegasus.she.stdfm.certification import STDFMCertificationPolicy


class CTRCertificationError(ValueError):
    """Raised when a CTR output is promoted without valid certification (§10 abort)."""


@dataclass(frozen=True)
class CTRCertificationPolicy:
    verified_mape: float = 0.15
    fragile_mape: float = 0.35
    verified_variance_ratio: float = 0.25
    fragile_variance_ratio: float = 0.40
    minimum_stability: float = 0.85
    constraint_residual_tol: float = 1e-3

    @classmethod
    def from_stdfm(cls, policy: STDFMCertificationPolicy | None = None) -> "CTRCertificationPolicy":
        policy = policy or STDFMCertificationPolicy()
        return cls(
            verified_mape=policy.verified_mape,
            fragile_mape=policy.fragile_mape,
            verified_variance_ratio=policy.verified_variance_ratio,
            fragile_variance_ratio=policy.fragile_variance_ratio,
            minimum_stability=policy.minimum_factor_stability,
        )


@dataclass(frozen=True)
class CTRCertificationRow:
    reconstruction_id: str
    status: str  # verified | fragile | illegal_excluded | uncertified
    holdout_mape: float | None
    reconstruction_var: float | None
    constraint_residual: float | None
    stability: float | None
    uncertainty_declared: bool
    warnings: tuple[str, ...] = ()

    @property
    def dashboard_safe(self) -> bool:
        return self.status == "verified" and self.uncertainty_declared

    @property
    def substrate_class(self) -> str:
        # CTR outputs are never B_official; latent factors are B_latent, everything
        # else reconstructed.
        return "illegal_excluded" if self.status == "illegal_excluded" else "B_reconstructed"

    def as_manifest(self) -> dict[str, Any]:
        return {
            "reconstruction_id": self.reconstruction_id,
            "status": self.status,
            "holdout_mape": self.holdout_mape,
            "reconstruction_var": self.reconstruction_var,
            "constraint_residual": self.constraint_residual,
            "stability": self.stability,
            "uncertainty_declared": self.uncertainty_declared,
            "dashboard_safe": self.dashboard_safe,
            "substrate_class": self.substrate_class,
            "warnings": list(self.warnings),
        }


def certify_ctr(
    *,
    reconstruction_id: str,
    holdout_mape: float | None,
    reconstruction_var: float | None,
    constraint_residual: float | None,
    stability: float | None,
    uncertainty_declared: bool,
    warnings: tuple[str, ...] = (),
    policy: CTRCertificationPolicy | None = None,
) -> CTRCertificationRow:
    policy = policy or CTRCertificationPolicy()

    def row(status: str, extra: tuple[str, ...] = ()) -> CTRCertificationRow:
        return CTRCertificationRow(
            reconstruction_id=reconstruction_id,
            status=status,
            holdout_mape=holdout_mape,
            reconstruction_var=reconstruction_var,
            constraint_residual=constraint_residual,
            stability=stability,
            uncertainty_declared=uncertainty_declared,
            warnings=tuple(dict.fromkeys([*warnings, *extra])),
        )

    if constraint_residual is not None and constraint_residual > policy.constraint_residual_tol:
        return row("illegal_excluded", ("ctr_constraint_residual_exceeds_tolerance",))
    if holdout_mape is None or reconstruction_var is None or stability is None:
        return row("uncertified", ("ctr_metrics_incomplete",))
    if (
        holdout_mape <= policy.verified_mape
        and reconstruction_var <= policy.verified_variance_ratio
        and stability >= policy.minimum_stability
    ):
        return row("verified")
    if holdout_mape <= policy.fragile_mape and reconstruction_var <= policy.fragile_variance_ratio:
        return row("fragile")
    return row("illegal_excluded", ("ctr_failed_certification",))


def assert_ctr_verified_promotion_allowed(row: CTRCertificationRow) -> None:
    """§10 abort: reconstructed tensor promoted to verified without holdout/uncertainty."""
    if row.status != "verified":
        raise CTRCertificationError(
            f"CTR verified promotion requires status='verified' (got {row.status!r})."
        )
    if not row.uncertainty_declared:
        raise CTRCertificationError(
            "reconstructed tensor promoted to verified without propagated uncertainty."
        )


__all__ = [
    "CTRCertificationPolicy",
    "CTRCertificationRow",
    "CTRCertificationError",
    "certify_ctr",
    "assert_ctr_verified_promotion_allowed",
]
