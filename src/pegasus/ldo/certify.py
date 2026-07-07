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

from pegasus.ldo.multiplicity import benjamini_hochberg, fisher_z_pvalue
from pegasus.ldo.records import LinkRecord

_FDR_METHOD = "benjamini_hochberg_fisherz_neff"


class LDOCertificationError(ValueError):
    """Raised when a dependency edge is promoted without valid certification (§10)."""


@dataclass(frozen=True)
class LDOCertificationPolicy:
    min_stability: float = 0.6
    require_uncertainty_for_nonlinear: bool = True
    fdr_q: float = 0.1
    fdr_enforce: bool = False   # default: annotate q only, leave certification_status untouched


def certify_link(record: LinkRecord, *, policy: LDOCertificationPolicy | None = None) -> LinkRecord:
    """Return the record with a certification_status reflecting the gate."""
    policy = policy or LDOCertificationPolicy()

    # Low-power edges are already marked descriptive upstream; keep them so.
    if "low_n_eff_descriptive_only" in record.warnings:
        return replace(record, certification_status="descriptive")

    # §V.6: an unconverged low-rank ADMM solve yields an unreliable S/L split — never
    # present its edges as exact. Downgrade to descriptive (surfaced, not certified).
    if "lowrank_unconverged_descriptive_only" in record.warnings:
        return replace(record, certification_status="descriptive")

    # §III.8 conjuncts (certgates.py): an edge that survives only a single λ operating point
    # (regularization-path disagreement) or a directed lag whose endpoints share a latent factor
    # (latent-vs-lag confound, §IX.2) is not a certifiable discovery — surface descriptive.
    if any(w.startswith("low_regularization_path_agreement") for w in record.warnings):
        return replace(record, certification_status="descriptive")
    if "possible_latent_lag_confound" in record.warnings:
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

    # lagged_directed / contemporaneous (§III.8): promotion is a CONJUNCTION — holdout
    # stability above threshold AND propagated uncertainty. An edge with either missing is
    # surfaced descriptive, never certified as a discovery.
    stab = record.stability
    if stab is None or stab < policy.min_stability:
        return replace(record, certification_status="descriptive")
    if record.uncertainty is None:
        return replace(record, certification_status="descriptive",
                       warnings=record.warnings + ("uncertified_missing_propagated_uncertainty",))
    return replace(record, certification_status="selected")


def certify_links(records: list[LinkRecord], *, policy: LDOCertificationPolicy | None = None) -> list[LinkRecord]:
    policy = policy or LDOCertificationPolicy()
    gated = [certify_link(r, policy=policy) for r in records]
    return _apply_fdr(gated, policy)


def _apply_fdr(records: list[LinkRecord], policy: LDOCertificationPolicy) -> list[LinkRecord]:
    """BH across edges with a testable partial-correlation at their effective-n. Default is
    annotate-only (q + fdr_method); fdr_enforce additionally downgrades a *selected* edge that
    fails FDR to descriptive. Untestable edges (no n_eff / no pcorr) are skipped (q=None)."""
    idx = [
        i for i, r in enumerate(records)
        if r.n_eff is not None and r.partial_correlation is not None
    ]
    if not idx:
        return records
    pvals = [fisher_z_pvalue(records[i].partial_correlation, records[i].n_eff) for i in idx]
    rejected, qvals = benjamini_hochberg(pvals, policy.fdr_q)
    out = list(records)
    for j, i in enumerate(idx):
        r = out[i]
        r = replace(r, fdr_qvalue=qvals[j], fdr_method=_FDR_METHOD)
        if policy.fdr_enforce and r.certification_status == "selected" and not rejected[j]:
            r = replace(r, certification_status="descriptive",
                        warnings=r.warnings + ("fdr_not_significant_descriptive_only",))
        out[i] = r
    return out


def assert_ldo_edge_promotion_allowed(record: LinkRecord) -> None:
    """§III.8/§10 standing abort: a promoted (``selected``) edge MUST carry holdout stability
    AND propagated uncertainty. The final backstop over every record on the live path — it
    hard-fails rather than relying only on ``certify_link``'s soft downgrade."""
    if record.certification_status != "selected":
        return
    if record.edge_type in {"lagged_directed", "contemporaneous"}:
        if record.stability is None or record.uncertainty is None:
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
