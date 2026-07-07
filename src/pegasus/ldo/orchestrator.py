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

from pegasus.ldo.assemble import LDOField, assemble_ldo_tensor
from pegasus.ldo.certify import LDOCertificationPolicy, certify_links
from pegasus.ldo.envelope import assert_within_envelope
from pegasus.ldo.edges import (
    annotate_disease_provenance,
    stability_select,
    to_link_records,
    type_mechanical_overlap,
)
from pegasus.ldo.lags import fit_lagged_links
from pegasus.ldo.margins import GaussianField, gaussianize_field
from pegasus.ldo.records import LinkRecord
from pegasus.ldo.residual_scan import scan_residual_nonlinear_edges
from pegasus.she.panel import CommonPanel


@dataclass
class LDORun:
    link_records: list[LinkRecord]
    variables: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def hypotheses_rows(self) -> list[dict[str, Any]]:
        return [r.as_row() for r in self.link_records]


def _prepare_ldo_inputs(
    source: CommonPanel | LDOField | GaussianField,
    *,
    seed: int,
    keep_variables: set[str] | frozenset[str] | None = None,
    exposure=None,
) -> tuple[LDOField | None, GaussianField]:
    """Return ``(raw_field, gaussian_field)``. The raw (pre-gaussianized) LDOField is
    kept for causal orientation — LiNGAM cannot identify direction on gaussianized
    data. It is ``None`` when the source is already a GaussianField (no raw values)."""
    if isinstance(source, CommonPanel):
        raw = assemble_ldo_tensor(source, keep_variables=keep_variables)
        return raw, gaussianize_field(raw, seed=seed, exposure=exposure)
    if isinstance(source, LDOField):
        return source, gaussianize_field(source, seed=seed, exposure=exposure)
    if isinstance(source, GaussianField):
        return None, source
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
    gamma_temporal: float = 0.1,
    gamma_disease: float = 0.1,
    variable_meta: dict[str, dict] | None = None,
    exposure=None,
) -> LDORun:
    """Fit the LDO and read off certified LinkRecords in one pass.

    ``disease_graph`` (a structural ``DiseaseGraph``, §II.6) supplies the disease-axis
    prior: related disease-concept variables get a lower ℓ1 penalty so their sparse
    links survive. Absent it, the estimator is the plain scalar-penalty LVGLASSO.
    """
    raw_field, gf = _prepare_ldo_inputs(source, seed=seed, keep_variables=keep_variables, exposure=exposure)
    p, S, T = gf.shape

    requested_K = K
    if adaptive_k:
        K = _adaptive_lag_order(K, p=p, T=T)

    # §II.10: refuse a run whose dense form exceeds the compute envelope rather
    # than silently subsampling; the caller should tile/multi-resolve (§II.7).
    envelope_bytes = assert_within_envelope(p=p, S=S, T=T, K=K) if enforce_envelope else None

    disease_penalty = None
    disease_laplacian = None
    if disease_graph is not None:
        from pegasus.ldo.disease_prior import disease_laplacian_matrix, disease_penalty_matrix
        disease_penalty = disease_penalty_matrix(gf.variables, disease_graph, lambda1=lambda1)
        # §III.4(5) disease-Laplacian quadratic operand (smooth precision rows across the
        # CID-10 hierarchy). Distinct from the ℓ1 penalty above (edge selection); this is the
        # GMRF/hierarchy quadratic. Absent a disease graph, only temporal smoothing applies.
        disease_laplacian = disease_laplacian_matrix(gf.variables, disease_graph)

    fit_kwargs = dict(
        kappa=kappa, lambda1=lambda1, lambda2=lambda2,
        edge_threshold=edge_threshold, disease_penalty=disease_penalty,
        disease_laplacian=disease_laplacian,
        gamma_temporal=gamma_temporal, gamma_disease=gamma_disease,
    )
    lagged = fit_lagged_links(gf, K=K, **fit_kwargs)

    stability = stability_select(
        gf, K=K, n_subsamples=n_subsamples, subsample_frac=subsample_frac, seed=seed + 1,
        max_workers=max_workers, **fit_kwargs
    )
    records = to_link_records(
        lagged, field=gf, stability=stability, stability_threshold=stability_threshold,
    )

    # Disease-axis provenance + the mandatory shared-code overlap guard (§5.3): a link
    # between concept-variables built on overlapping codes is mechanical, not a discovery.
    if variable_meta:
        records = annotate_disease_provenance(records, variable_meta)
        code_sets = {v: frozenset(m["code_set"]) for v, m in variable_meta.items() if m.get("code_set")}
        if code_sets:
            records = type_mechanical_overlap(records, code_sets)

    if run_residual_scan:
        # Residual scan uses the lag-0 precision block, aligned p×p to gf.variables
        # (built through the kept-feature map in lags.py; NOT the raw S[:p,:p] slice,
        # which misindexes when low-coverage variables are dropped).
        lag0 = lagged.lag0_precision
        if lag0 is None:
            lag0 = np.eye(p)
        try:
            records.extend(
                scan_residual_nonlinear_edges(gf, lag0, budget=budget, seed=seed + 2)
            )
            _residual_error = None
        except Exception as exc:  # residual scan is additive; never fail the run
            _residual_error = f"{type(exc).__name__}:{exc}"
    else:
        _residual_error = None

    # §V.6 convergence gate: if the low-rank ADMM did not converge, the S/L split (and
    # every edge read off it, backbone and residual) is unreliable — flag so the
    # certifier downgrades to descriptive rather than certifying an approximation as exact.
    if not lagged.fit.converged:
        from dataclasses import replace as _replace
        records = [
            _replace(r, warnings=tuple(r.warnings) + ("lowrank_unconverged_descriptive_only",))
            for r in records
        ]

    records = certify_links(records, policy=certification_policy)

    # Rung-1 causal orientation (§IV): direct contemporaneous edges by non-Gaussian
    # LiNGAM where identifiable (lagged edges already carry time precedence); the LDO
    # emits an oriented hypothesis skeleton, not just undirected associations. Orientation
    # needs the RAW (non-gaussianized) values — direction is unidentifiable on Gaussian
    # data — so it is a no-op (all undirected) when the source was a GaussianField.
    from pegasus.causal.orient import orient_links
    if raw_field is not None:
        orient_data = {v: raw_field.X[i].reshape(-1) for i, v in enumerate(raw_field.variables)}
    else:
        orient_data = {v: gf.Z[i].reshape(-1) for i, v in enumerate(gf.variables)}
    records = orient_links(records, orient_data)

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
        # Disease-axis effects (visible only when variable_meta/disease_graph were threaded):
        # the mechanical-overlap guard's re-typings and the disease-informed penalty.
        "disease_prior_applied": disease_penalty is not None,
        "n_mechanical_overlap": sum(1 for r in records if r.edge_type == "mechanical_overlap"),
        "n_disease_provenanced": sum(1 for r in records if r.code_system is not None),
        # Rung-1 orientation (§IV): edges given a non-Gaussian LiNGAM direction vs left undirected.
        "n_oriented_lingam": sum(1 for r in records if "oriented_non_gaussian_lingam" in r.warnings),
        "n_orientation_undirected": sum(1 for r in records if "orientation_undirected_unidentifiable" in r.warnings),
    }
    return LDORun(link_records=records, variables=gf.variables, diagnostics=diagnostics)


__all__ = ["LDORun", "run_ldo"]
