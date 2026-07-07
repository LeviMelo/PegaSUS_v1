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

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pegasus.ldo.assemble import LDOField, assemble_ldo_tensor
from pegasus.ldo.certify import LDOCertificationPolicy, certify_links
from pegasus.ldo.envelope import ScaleExceedsEnvelopeError, assert_within_envelope
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
    exposure_field_by_variable=None,
    measured_quantity_by_variable=None,
) -> tuple[LDOField | None, GaussianField]:
    """Return ``(raw_field, gaussian_field)``. The raw (pre-gaussianized) LDOField is
    kept for causal orientation — LiNGAM cannot identify direction on gaussianized
    data. It is ``None`` when the source is already a GaussianField (no raw values).

    ``exposure_field_by_variable`` maps a count variable to its denominator field in the
    panel; the assembled per-variable exposure tensor drives the §III.5 count-with-exposure
    margin (extensive counts modelled net of exposure), overriding an explicit ``exposure``."""
    if isinstance(source, CommonPanel):
        raw = assemble_ldo_tensor(source, keep_variables=keep_variables,
                                  exposure_field_by_variable=exposure_field_by_variable,
                                  measured_quantity_by_variable=measured_quantity_by_variable)
        exp = raw.exposure if raw.exposure is not None else exposure
        return raw, gaussianize_field(raw, seed=seed, exposure=exp)
    if isinstance(source, LDOField):
        exp = source.exposure if source.exposure is not None else exposure
        return source, gaussianize_field(source, seed=seed, exposure=exp)
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
    precision_target: float | None = None,
    precision_budget: float = 1.0,
    variable_meta: dict[str, dict] | None = None,
    exposure=None,
    exposure_field_by_variable=None,
    measured_quantity_by_variable=None,
    float32_bulk: bool | str = "auto",
    spatial_field_dir: str | None = None,
    max_spatial_fields: int = 24,
) -> LDORun:
    """Fit the LDO and read off certified LinkRecords in one pass.

    ``disease_graph`` (a structural ``DiseaseGraph``, §II.6) supplies the disease-axis
    prior: related disease-concept variables get a lower ℓ1 penalty so their sparse
    links survive. Absent it, the estimator is the plain scalar-penalty LVGLASSO.
    """
    raw_field, gf = _prepare_ldo_inputs(source, seed=seed, keep_variables=keep_variables,
                                        exposure=exposure, exposure_field_by_variable=exposure_field_by_variable,
                                        measured_quantity_by_variable=measured_quantity_by_variable)
    p, S, T = gf.shape

    requested_K = K
    if adaptive_k:
        K = _adaptive_lag_order(K, p=p, T=T)

    # §V.1 precision policy: "auto" stores the ADMM bulk iterates in float32 (half the working
    # set, float64 reductions + condition-number escalation) once the float64 fit would use more
    # than half the envelope — i.e. under memory pressure — and stays float64 otherwise for
    # maximum numerical fidelity. Small/test runs are unaffected (stay float64).
    if float32_bulk == "auto":
        from pegasus.ldo.envelope import estimate_ldo_bytes, load_compute_envelope
        env = load_compute_envelope()
        f32 = estimate_ldo_bytes(p=p, S=S, T=T, K=K, float32_bulk=False) > 0.5 * env.max_bytes
    else:
        f32 = bool(float32_bulk)

    # §II.10: refuse a run whose dense form exceeds the compute envelope rather
    # than silently subsampling; the caller should tile/multi-resolve (§II.7). The guard's
    # byte model matches the fit's actual working dtype (float32_bulk).
    envelope_bytes = assert_within_envelope(p=p, S=S, T=T, K=K, float32_bulk=f32) if enforce_envelope else None

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
        float32_bulk=f32,
    )
    lagged = fit_lagged_links(gf, K=K, **fit_kwargs)

    # §V.5 Adaptive Precision Controller: uncertainty drives compute. Model the low-rank
    # readout as a decision-relevant Quantity (statistical = Fisher-z SE at n_eff, numerical =
    # the randomized-SVD truncation error); if the numerical error DOMINATES and the budget
    # allows, escalate the readout to EXACT (re-fit with a dense eigh) — else leave the
    # data-limited fit alone. Report met vs approximation-limited (typed, never silent). Off
    # (precision_target=None) → no controller, no behaviour change.
    precision_report = None
    if precision_target is not None:
        import math as _math

        from pegasus.compute.controller import Budget, Quantity, adaptive_precision_run
        n_eff0 = int(np.isfinite(gf.Z).any(axis=0).sum())
        stat_unc = 1.0 / _math.sqrt(max(n_eff0 - 3, 1))
        q = Quantity(name="ldo_lowrank_readout", statistical_uncertainty=stat_unc,
                     numerical_uncertainty=float(lagged.fit.numerical_error), exact_cost=1.0)
        report = adaptive_precision_run([q], target=float(precision_target), budget=Budget(total=float(precision_budget)))
        if report.spent > 0:  # controller escalated → exact low-rank readout
            lagged = fit_lagged_links(gf, K=K, randomized_factors=False, **fit_kwargs)
        precision_report = {
            "target": float(precision_target), "spent": report.spent,
            "met": list(report.met), "approximation_limited": list(report.approximation_limited),
            "data_limited": list(report.data_limited), "escalated_to_exact": report.spent > 0,
        }

    stability = stability_select(
        gf, K=K, n_subsamples=n_subsamples, subsample_frac=subsample_frac, seed=seed + 1,
        max_workers=max_workers, **fit_kwargs
    )
    records = to_link_records(
        lagged, field=gf, stability=stability, stability_threshold=stability_threshold,
        numerical_error=lagged.fit.numerical_error,
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
        except ScaleExceedsEnvelopeError:
            # §II.10 / §V.1: a compute-envelope refusal is a LOUD, typed refusal — it must
            # propagate, not be silently downgraded to a diagnostic string (that would let a
            # national run quietly skip the scan it cannot afford instead of refusing).
            raise
        except Exception as exc:  # a genuine scan bug is additive; the backbone edges still stand
            _residual_error = f"{type(exc).__name__}:{exc}"
    else:
        _residual_error = None

    # §V.6 convergence gate: if the low-rank ADMM did not converge, the S/L split (and
    # every edge read off it, backbone and residual) is unreliable — flag so the
    # certifier downgrades to descriptive rather than certifying an approximation as exact.
    if not lagged.fit.converged:
        records = [
            replace(r, warnings=tuple(r.warnings) + ("lowrank_unconverged_descriptive_only",))
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

    # Rung-2 quasi-experimental escalation (§IV): for directed edges whose target series shows
    # a detected structural break, run an interrupted-time-series and promote to Rung 2 with the
    # ITS evidence. Machine-checkable auto-trigger; a validated external shock is expert refinement.
    if T >= 10:
        from pegasus.causal.quasi import escalate_rung2_its
        series_by_var = {v: np.nanmean(gf.Z[i], axis=0) for i, v in enumerate(gf.variables)}
        records = escalate_rung2_its(records, series_by_var)

    # §III.3/§III.7 spatial BYM varying-coefficient field: for the strongest *selected* directed
    # / contemporaneous edges, fit the per-locality slope field β_s (GMRF-smoothed) and persist it
    # as the edge's spatial-heterogeneity sidecar, setting spatial_field_ref. Opt-in (a dir must be
    # given); bounded to max_spatial_fields; best-effort per edge. This is the effect-modification
    # surface — "is X→Y stronger where …" — read at national/region/state/muni scales.
    n_spatial_fields = 0
    if spatial_field_dir is not None and gf.shape[1] > 1:
        from pathlib import Path as _Path

        from pegasus.ldo.spatial_field import fit_spatial_varying_coefficient, write_spatial_field
        _sdir = _Path(spatial_field_dir)
        _sdir.mkdir(parents=True, exist_ok=True)
        _cands = [
            (i, r) for i, r in enumerate(records)
            if r.edge_type in ("lagged_directed", "contemporaneous")
            and r.certification_status == "selected"
        ]
        _cands.sort(key=lambda ir: abs(ir[1].weight or 0.0), reverse=True)
        for i, r in _cands[:max_spatial_fields]:
            try:
                sf = fit_spatial_varying_coefficient(gf, r.source_var, r.target_var, kappa=kappa)
                if sf is None:
                    continue
                _safe = f"{r.source_var}__{r.target_var}__lag{r.lag_k}".replace("/", "_")
                ref = write_spatial_field(sf, _sdir / f"{_safe}.spatial_field.parquet")
                records[i] = replace(r, spatial_field_ref=ref)
                n_spatial_fields += 1
            except Exception:
                continue

    # §III.8 standing-abort backstop over every final record: a promoted edge without holdout
    # stability AND propagated uncertainty hard-fails rather than shipping an uncertifiable claim.
    from pegasus.ldo.certify import assert_ldo_edge_promotion_allowed
    for r in records:
        assert_ldo_edge_promotion_allowed(r)

    # §V.2 separable joint-precision telemetry (§V.6): the factored log-det of the joint
    # operator Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹, computed axis-by-axis — the (pST)² joint is never
    # materialized. Exact sparse-Cholesky space factor at moderate S; matrix-free SLQ at
    # national S. Best-effort telemetry feeding the validity report; never fails a run.
    kronecker_report = None
    try:
        if lagged.lag0_precision is not None and gf.shape[1] > 1:
            from pegasus.ldo.kron import joint_logdet, kronecker_from_ldo
            _op = kronecker_from_ldo(lagged.lag0_precision, gf.space_ids, kappa=kappa, tau=K + 1)
            _ld, _method = joint_logdet(_op, seed=seed)
            kronecker_report = {
                "joint_logdet": _ld, "space_logdet_method": _method,
                "dim": _op.dim, "p": _op.p, "S": _op.S, "tau": _op.tau,
            }
    except Exception:
        kronecker_report = None

    n_eff = int(np.isfinite(gf.Z).any(axis=0).sum())
    # §VIII.2(3) typed coverage manifest: record what was searched (resolution, lag depth,
    # functional forms) and, explicitly, what was not — so "no edge" ≠ "not looked for".
    from pegasus.ldo.coverage import build_coverage_manifest
    coverage = build_coverage_manifest(
        resolution=str(getattr(gf, "resolution", "cell")), K=K, requested_K=requested_K,
        ran_residual_scan=run_residual_scan, residual_error=_residual_error,
    )
    diagnostics = {
        "p": p, "S": S, "T": T, "K": K, "K_requested": requested_K,
        "n_eff": n_eff,
        "lowrank_factors": int(lagged.fit.factor_loadings.shape[1]),
        "lowrank_converged": bool(lagged.fit.converged),
        "n_link_records": len(records),
        "n_selected": sum(1 for r in records if r.certification_status == "selected"),
        "residual_scan_error": _residual_error,
        "envelope_bytes": envelope_bytes,
        # §V.1/§V.4 precision policy: the ADMM bulk dtype actually used, plus whether an
        # ill-conditioned covariance forced a float32→float64 escalation (self-correcting).
        "precision_policy": {
            "float32_bulk_requested": bool(f32),
            "work_dtype": str(getattr(lagged.fit, "work_dtype", "float64")),
            "cond_escalated": bool(getattr(lagged.fit, "cond_escalated", False)),
        },
        # Disease-axis effects (visible only when variable_meta/disease_graph were threaded):
        # the mechanical-overlap guard's re-typings and the disease-informed penalty.
        "disease_prior_applied": disease_penalty is not None,
        "n_mechanical_overlap": sum(1 for r in records if r.edge_type == "mechanical_overlap"),
        "n_disease_provenanced": sum(1 for r in records if r.code_system is not None),
        # §IV causal ladder: edges by rung (0 associational / 1 oriented LiNGAM+collider /
        # 2 quasi-experimental ITS). Rung 3 is expert-invoked only, never autonomous.
        "n_oriented_lingam": sum(1 for r in records if "oriented_non_gaussian_lingam" in r.warnings),
        "n_oriented_collider": sum(1 for r in records if "oriented_collider" in r.warnings),
        "n_orientation_undirected": sum(1 for r in records if "orientation_undirected_unidentifiable" in r.warnings),
        "n_rung0": sum(1 for r in records if (r.causal_rung or 0) == 0),
        "n_rung1": sum(1 for r in records if r.causal_rung == 1),
        "n_rung2": sum(1 for r in records if r.causal_rung == 2),
        # §III.5: number of variables modelled with the count-with-exposure (Poisson-offset)
        # margin rather than the plain rank margin (extensive counts net of exposure).
        "count_exposure_variables": (
            int(np.isfinite(raw_field.exposure).any(axis=(1, 2)).sum())
            if raw_field is not None and getattr(raw_field, "exposure", None) is not None else 0
        ),
        # §I.2/§III.5 audit-named alias: number of variables actually routed onto the Poisson-
        # offset (count-with-exposure) margin — i.e. extensive-numerator RN fields. Intensive
        # RN outputs (rate÷rate, densities) carry no measured_quantity_ref (kernels.py gate) and
        # stay on the rank-PIT margin, so this counts only genuine extensive counts.
        "n_exposure_margins": (
            int((np.isfinite(raw_field.exposure) & (raw_field.exposure > 0)).any(axis=(1, 2)).sum())
            if raw_field is not None and getattr(raw_field, "exposure", None) is not None else 0
        ),
        # §V.6(2): mean propagated numerical (randomized-SVD) error folded into edge uncertainty.
        "numerical_error": float(lagged.fit.numerical_error),
        # §VIII.2(3): typed coverage manifest (searched + explicitly-unsearched regions).
        "coverage_manifest": coverage.as_manifest(),
        # §V.5: adaptive-precision-controller verdict (None when the controller is off).
        "precision_controller": precision_report,
        # §V.2: separable joint-precision log-det via the Kronecker-factored operator (None
        # when the variable precision is not SPD or the space factor is unavailable).
        "kronecker_joint": kronecker_report,
        # §III.3/§III.7: number of edges given a fitted spatial BYM varying-coefficient field
        # (0 unless spatial_field_dir was provided → edges carry spatial_field_ref).
        "n_spatial_fields": n_spatial_fields,
    }
    return LDORun(link_records=records, variables=gf.variables, diagnostics=diagnostics)


def run_ldo_multiresolution(
    source: CommonPanel | LDOField | GaussianField,
    *,
    K: int = 8,
    coarse_level: int = 2,
    coarse_K: int | None = None,
    seed: int = 0,
    keep_variables: set[str] | frozenset[str] | None = None,
    exposure=None,
    exposure_field_by_variable=None,
    measured_quantity_by_variable=None,
    **ldo_kwargs,
) -> LDORun:
    """§II.7 / RES-01 coarse→fine LDO on a panel/field source — a drop-in for :func:`run_ldo`.

    Prepares the fine (municipality-grain) gaussian field once, spatially coarsens it to a cheap
    discovery grain (cod6 prefix ``coarse_level``: 2 = UF, 1 = macroregion), and runs the two-pass
    :func:`run_multiresolution_ldo` — a §VIII.2(1) recall-tuned sensitivity screen picks candidate
    pairs from the coarse pass, only those variables are refit at fine grain, and a §VIII.2(2)
    random deep audit of the pruned variables measures the false-negative rate. Returns a merged
    ``LDORun`` (fine-sharpened candidate edges + coarse-only edges) with the fine pass's rich
    diagnostics plus a ``multiresolution`` summary, so callers consume it exactly like ``run_ldo``.

    The count-with-exposure / explicit-exposure inputs are consumed once when the fine field is
    gaussianized here; they are not re-forwarded to the per-grain fits (which take the prepared
    field). This is the compute strategy for fine grains too large to fit wholesale (§V.1): each
    pass is envelope-sized (coarse ``S≈27``; fine restricted to candidate variables).
    """
    from pegasus.ldo.resolution import coarsen_field_spatial, run_multiresolution_ldo

    _, fine = _prepare_ldo_inputs(
        source, seed=seed, keep_variables=keep_variables, exposure=exposure,
        exposure_field_by_variable=exposure_field_by_variable,
        measured_quantity_by_variable=measured_quantity_by_variable,
    )
    coarse = coarsen_field_spatial(fine, level=coarse_level)
    ck = coarse_K if coarse_K is not None else max(1, min(K, 4))
    mr = run_multiresolution_ldo(coarse, fine, coarse_K=ck, fine_K=K, seed=seed, **ldo_kwargs)
    base = mr.fine.diagnostics if mr.fine is not None else mr.coarse.diagnostics
    diagnostics = dict(base)
    diagnostics["multiresolution"] = mr.diagnostics
    return LDORun(link_records=mr.link_records, variables=fine.variables, diagnostics=diagnostics)


__all__ = ["LDORun", "run_ldo"]
