"""LDO orchestrator — one in-memory run (MSD-II §II.6, MII-LDO-06).

``run_ldo`` executes the whole Lattice Dependency Operator as one call:
margins → structured precision + low-rank (with lag extension) → stability
selection → link readout → residual nonlinear (HSIC) scan → certification. There
are no inter-stage manifests written to disk — the slice-zoo's ``Tables/pirs_*``
hand-offs are replaced by in-memory phases here.

Input may be a compiled ``CommonPanel``, an assembled ``LDOField``, or a
``GaussianField``; the orchestrator advances it to the latent Gaussian scale as
needed. Output is a typed ``LinkRecord`` list (the enriched ``Hypotheses`` key,
§II.8) plus diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pegasus.pirs.ldo.assemble import LDOField, assemble_ldo_tensor
from pegasus.pirs.ldo.certify import LDOCertificationPolicy, certify_links
from pegasus.pirs.ldo.envelope import assert_within_envelope
from pegasus.pirs.ldo.edges import stability_select, to_link_records
from pegasus.pirs.ldo.lags import fit_lagged_links
from pegasus.pirs.ldo.margins import GaussianField, gaussianize_field
from pegasus.pirs.ldo.records import LinkRecord
from pegasus.pirs.ldo.residual_scan import scan_residual_nonlinear_edges
from pegasus.she.panel import CommonPanel


@dataclass
class LDORun:
    link_records: list[LinkRecord]
    variables: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def hypotheses_rows(self) -> list[dict[str, Any]]:
        return [r.as_row() for r in self.link_records]


def _to_gaussian_field(
    source: CommonPanel | LDOField | GaussianField,
    *,
    seed: int,
    keep_variables: set[str] | frozenset[str] | None = None,
) -> GaussianField:
    if isinstance(source, CommonPanel):
        return gaussianize_field(assemble_ldo_tensor(source, keep_variables=keep_variables), seed=seed)
    if isinstance(source, LDOField):
        return gaussianize_field(source, seed=seed)
    if isinstance(source, GaussianField):
        return source
    raise TypeError(f"run_ldo cannot consume {type(source).__name__}")


def _adaptive_lag_order(requested_K: int, *, p: int, T: int) -> int:
    """Cap the lag order to what the panel can support and afford.

    A lag-``k`` link needs ``k < T`` distinct time points (``fit_lagged_links``
    requires ``T > K``), so annual multi-year panels can't carry K=8. And the
    precision solve is ``O((p·(K+1))^3)`` — for a context-heavy panel (large ``p``)
    every extra lag is expensive and context associations are overwhelmingly
    cross-sectional, so shrink K as ``p`` grows. Never increases the request.
    """
    k = max(1, min(requested_K, T - 2)) if T > 2 else max(1, min(requested_K, 1))
    if p > 200:
        k = min(k, 1)
    elif p > 130:
        k = min(k, 2)
    return k


def run_ldo(
    source: CommonPanel | LDOField | GaussianField,
    *,
    K: int = 3,
    kappa: float = 1.0,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    edge_threshold: float = 0.05,
    n_subsamples: int = 12,
    subsample_frac: float = 0.7,
    stability_threshold: float = 0.6,
    budget: str = "standard",
    run_residual_scan: bool = True,
    seed: int = 0,
    certification_policy: LDOCertificationPolicy | None = None,
    enforce_envelope: bool = True,
    keep_variables: set[str] | frozenset[str] | None = None,
    max_workers: int | None = None,
    adaptive_k: bool = True,
    disease_graph=None,
) -> LDORun:
    """Fit the LDO and read off certified LinkRecords in one pass.

    ``disease_graph`` (a structural ``DiseaseGraph``, §II.6) supplies the disease-axis
    prior: related disease-concept variables get a lower ℓ1 penalty so their sparse
    links survive. Absent it, the estimator is the plain scalar-penalty LVGLASSO.
    """
    gf = _to_gaussian_field(source, seed=seed, keep_variables=keep_variables)
    p, S, T = gf.shape

    requested_K = K
    if adaptive_k:
        K = _adaptive_lag_order(K, p=p, T=T)

    # §II.10: refuse a run whose dense form exceeds the compute envelope rather
    # than silently subsampling; the caller should tile/multi-resolve (§II.7).
    envelope_bytes = assert_within_envelope(p=p, S=S, T=T, K=K) if enforce_envelope else None

    disease_penalty = None
    if disease_graph is not None:
        from pegasus.pirs.ldo.disease_prior import disease_penalty_matrix
        disease_penalty = disease_penalty_matrix(gf.variables, disease_graph, lambda1=lambda1)

    fit_kwargs = dict(
        kappa=kappa, lambda1=lambda1, lambda2=lambda2,
        edge_threshold=edge_threshold, disease_penalty=disease_penalty,
    )
    lagged = fit_lagged_links(gf, K=K, **fit_kwargs)

    stability = stability_select(
        gf, K=K, n_subsamples=n_subsamples, subsample_frac=subsample_frac, seed=seed + 1,
        max_workers=max_workers, **fit_kwargs
    )
    records = to_link_records(
        lagged, field=gf, stability=stability, stability_threshold=stability_threshold,
    )

    if run_residual_scan:
        # Residual scan uses the contemporaneous (lag-0) precision block.
        lag0 = lagged.fit.S[:p, :p]
        try:
            records.extend(
                scan_residual_nonlinear_edges(gf, lag0, budget=budget, seed=seed + 2)
            )
        except Exception as exc:  # residual scan is additive; never fail the run
            records = records  # noqa: PLW0127
            _residual_error = f"{type(exc).__name__}:{exc}"
        else:
            _residual_error = None
    else:
        _residual_error = None

    records = certify_links(records, policy=certification_policy)

    n_eff = int(np.isfinite(gf.Z).any(axis=0).sum())
    diagnostics = {
        "p": p, "S": S, "T": T, "K": K, "K_requested": requested_K,
        "n_eff": n_eff,
        "lowrank_factors": int(lagged.fit.factor_loadings.shape[1]),
        "lowrank_converged": bool(lagged.fit.converged),
        "n_link_records": len(records),
        "n_selected": sum(1 for r in records if r.certification_status == "selected"),
        "residual_scan_error": _residual_error,
        "envelope_bytes": envelope_bytes,
    }
    return LDORun(link_records=records, variables=gf.variables, diagnostics=diagnostics)


__all__ = ["LDORun", "run_ldo"]
