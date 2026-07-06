"""LDO certification gate (MSD-II §II.6.4, MII-LDO-06).

Generalizes the CTR/ST-DFM certification discipline to dependency edges: an edge
is promoted only with holdout stability and propagated uncertainty. Directionality
is reported only where time (lagged edges) licenses it; contemporaneous edges are
undirected. Consistent with MSD §1.1/§12 the LDO **produces causal hypotheses; it
does not certify causality.**

New §10 abort: "dependency edge promoted without holdout stability or propagated
uncertainty."
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pegasus.ldo.records import LinkRecord


class LDOCertificationError(ValueError):
    """Raised when a dependency edge is promoted without valid certification (§10)."""


@dataclass(frozen=True)
class LDOCertificationPolicy:
    min_stability: float = 0.6
    require_uncertainty_for_nonlinear: bool = True


def certify_link(record: LinkRecord, *, policy: LDOCertificationPolicy | None = None) -> LinkRecord:
    """Return the record with a certification_status reflecting the gate."""
    policy = policy or LDOCertificationPolicy()

    # Low-power edges are already marked descriptive upstream; keep them so.
    if "low_n_eff_descriptive_only" in record.warnings:
        return replace(record, certification_status="descriptive")

    if record.edge_type == "mechanical_overlap":
        # Shared-code correlation is known structure, never an epidemiological discovery (§5.3).
        return replace(record, certification_status="descriptive")

    if record.edge_type == "latent_shared":
        # Shared-driver flags are structural, not promoted causal edges.
        return replace(record, certification_status=record.certification_status or "descriptive")

    if record.edge_type == "nonlinear_residual":
        certified = (record.uncertainty is not None) or (not policy.require_uncertainty_for_nonlinear)
        return replace(record, certification_status="selected" if certified else "descriptive")

    # lagged_directed / contemporaneous: require stability above threshold.
    stab = record.stability
    if stab is None or stab < policy.min_stability:
        return replace(record, certification_status="descriptive")
    return replace(record, certification_status="selected")


def certify_links(records: list[LinkRecord], *, policy: LDOCertificationPolicy | None = None) -> list[LinkRecord]:
    return [certify_link(r, policy=policy) for r in records]


def assert_ldo_edge_promotion_allowed(record: LinkRecord) -> None:
    """§10 abort: a promoted (selected) edge must carry holdout stability or uncertainty."""
    if record.certification_status != "selected":
        return
    if record.edge_type in {"lagged_directed", "contemporaneous"}:
        if record.stability is None:
            raise LDOCertificationError(
                "dependency edge promoted without holdout stability or propagated uncertainty."
            )
    if record.edge_type == "nonlinear_residual" and record.uncertainty is None:
        raise LDOCertificationError(
            "nonlinear residual edge promoted without propagated uncertainty."
        )


__all__ = [
    "LDOCertificationPolicy",
    "LDOCertificationError",
    "certify_link",
    "certify_links",
    "assert_ldo_edge_promotion_allowed",
]
