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
    field_weights=None,
) -> tuple[LDOField | None, GaussianField]:
    """Return ``(raw_field, gaussian_field)``. The raw (pre-gaussianized) LDOField is
    kept for causal orientation — LiNGAM cannot identify direction on gaussianized
    data. It is ``None`` when the source is already a GaussianField (no raw values).

    ``exposure_field_by_variable`` maps a count variable to its denominator field in the
    panel; the assembled per-variable exposure tensor drives the §III.5 count-with-exposure
    margin (extensive counts modelled net of exposure), overriding an explicit ``exposure``."""
    if isinstance(source, CommonPanel):
        raw = assemble_ldo_tensor(source, keep_variables=keep_variables,
                                  field_weights=field_weights,
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
    select_lambda: bool = False,
    lambda_grid=None,
    lambda_beta: float = 0.05,
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
    disease_scale_precisions: dict[str, float] | None = None,
    gamma_temporal: float = 0.1,
    gamma_disease: float = 0.1,
    temporal_whiten: bool = False,
    spatial_kernel: str = "contiguity",
    precision_target: float | None = None,
    precision_budget: float = 1.0,
    variable_meta: dict[str, dict] | None = None,
    exposure=None,
    exposure_field_by_variable=None,
    measured_quantity_by_variable=None,
    field_weights: dict | None = None,
    float32_bulk: bool | str = "auto",
    spatial_field_dir: str | None = None,
    max_spatial_fields: int = 24,
    certify_approximation: bool | str = "auto",
    certify_strict: bool = False,
) -> LDORun:
    """Fit the LDO and read off certified LinkRecords in one pass.

    ``disease_graph`` (a structural ``DiseaseGraph``, §II.6) supplies the disease-axis
    prior: related disease-concept variables get a lower ℓ1 penalty so their sparse
    links survive. Absent it, the estimator is the plain scalar-penalty LVGLASSO.
    """
    raw_field, gf = _prepare_ldo_inputs(source, seed=seed, keep_variables=keep_variables,
                                        exposure=exposure, exposure_field_by_variable=exposure_field_by_variable,
                                        measured_quantity_by_variable=measured_quantity_by_variable,
                                        field_weights=field_weights)
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

    # Theme-4 StARS: pick λ₁ as the smallest penalty (densest graph) whose subsample
    # selection instability stays ≤ lambda_beta, before the main fit. Default off →
    # lambda1 stays as passed (0.1), zero behaviour change.
    lambda_selection = None
    if select_lambda:
        from pegasus.ldo.lambda_select import select_lambda_stars
        grid = lambda_grid if lambda_grid is not None else np.geomspace(lambda1 / 5.0, lambda1 * 5.0, 7)
        lambda1, lam_report = select_lambda_stars(
            gf, grid, K=K, n_subsamples=n_subsamples, subsample_frac=subsample_frac,
            seed=seed + 7, beta=lambda_beta,
            fit_kwargs=dict(kappa=kappa, lambda2=lambda2, edge_threshold=edge_threshold),
        )
        lambda_selection = {
            "method": "stars", "lambda_star": lam_report.lambda_star, "beta": lambda_beta,
            "lambda_grid": list(lam_report.lambda_grid), "instability": list(lam_report.instability),
        }

    disease_penalty = None
    disease_laplacian = None
    disease_shrink_flags: tuple[dict, ...] = ()
    if disease_graph is not None:
        from pegasus.ldo.disease_prior import (
            disease_laplacian_matrix,
            disease_penalty_matrix,
            heavily_shrunk_blocks,
            sum_of_scales_disease_operator,
        )
        disease_penalty = disease_penalty_matrix(gf.variables, disease_graph, lambda1=lambda1)
        # §III.3 sum-of-scales disease GMRF (θ_leaf = μ_chapter+δ_block+δ_category+δ_leaf). The per-
        # scale shrinkage τ_level is DATA-ESTIMATED by empirical Bayes from the RAW field values — NOT
        # the Gaussianized gf.Z, whose per-variable standardization zeroes every disease mean and would
        # force τ²→1 (over-shrink) everywhere (verified) — so a block of genuinely-alike diseases
        # shrinks hard and a divergent block lets the data speak. This replaces the flat γ_disease=0.1
        # magic constant (LDO-DIS-MAGIC-06 / P2) with an auto-determined, data-driven strength. An
        # explicit disease_scale_precisions overrides with fixed τ; absent a raw field, the flat L_D.
        disease_laplacian = None
        if disease_scale_precisions:
            disease_laplacian = sum_of_scales_disease_operator(
                gf.variables, disease_graph, disease_scale_precisions)
        elif raw_field is not None:
            _dvals = {v: raw_field.X[i].reshape(-1) for i, v in enumerate(raw_field.variables)}
            disease_laplacian = sum_of_scales_disease_operator(
                gf.variables, disease_graph, adaptive=True, field_values_by_variable=_dvals)
            if disease_laplacian is not None:
                disease_shrink_flags = tuple(heavily_shrunk_blocks(_dvals, gf.variables, disease_graph))
        if disease_laplacian is not None:
            gamma_disease = 1.0  # per-scale τ baked into G_D
        else:
            disease_laplacian = disease_laplacian_matrix(gf.variables, disease_graph)

    fit_kwargs = dict(
        kappa=kappa, lambda1=lambda1, lambda2=lambda2,
        edge_threshold=edge_threshold, disease_penalty=disease_penalty,
        disease_laplacian=disease_laplacian,
        gamma_temporal=gamma_temporal, gamma_disease=gamma_disease,
        float32_bulk=f32, temporal_whiten=temporal_whiten, spatial_kernel=spatial_kernel,
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

    # §V.6(3) exact-certifies-approximate: when the national fit used the randomized low-rank
    # readout (the p > 2·factor_rank_cap regime), certify that approximation on a bounded, real
    # spatial slice by running the SAME pipeline exact vs approximate and comparing overlapping
    # edges within propagated bounds. Records the report (never silent); ``certify_strict`` makes
    # a disagreement a LOUD refusal. Off (skipped) for small p where the fit is already exact.
    approx_certification = None
    _want_certify = (p > 48) if certify_approximation == "auto" else bool(certify_approximation)
    if _want_certify and S >= 3:
        from pegasus.ldo.exact_certify import ApproximationRejectedError
        from pegasus.validation.holdout import certify_approximation_on_slice
        try:
            approx_certification = certify_approximation_on_slice(
                gf, K=K, fit_kwargs=fit_kwargs, seed=seed + 5, strict=certify_strict)
        except ApproximationRejectedError:
            raise  # §V.6(3) loud rejection in strict mode
        except Exception as exc:
            approx_certification = {"ran": False, "reason": f"{type(exc).__name__}", "certified": None}

    # §IX.3 temporal holdout: fit through the training window, verify persistence on the held-out
    # tail year — a real edge persists/predicts, a fluke evaporates. Records the persistence rate
    # (never asserts out-of-sample without measuring it). Gated on a time span long enough to split.
    holdout_report = None
    if T >= 6:
        try:
            from pegasus.validation.holdout import temporal_holdout
            holdout_report = temporal_holdout(gf, K=K, fit_kwargs=fit_kwargs, seed=seed + 6)
        except Exception as exc:
            holdout_report = {"ran": False, "reason": f"{type(exc).__name__}"}

    stability = stability_select(
        gf, K=K, n_subsamples=n_subsamples, subsample_frac=subsample_frac, seed=seed + 1,
        max_workers=max_workers, **fit_kwargs
    )
    # The live fit GMRF-whitens across space when S>1 (fit_lagged_links spatial_whiten default), so
    # spatial dependence is already removed from the correlation — the edge effective-n must NOT be
    # Moran-deflated again (Theme-6 double-correction). Only load/apply the structural graph's
    # spatial deflation for a NON-whitened fit; when whitened, edge n_eff is reliability-weighted only.
    _whitened = gf.shape[1] > 1
    _spatial_graph = None
    if not _whitened:
        try:
            from pegasus.geo.spatial_graph import load_spatial_graph, structural_graph_available
            if structural_graph_available():
                _spatial_graph = load_spatial_graph()
        except Exception:
            _spatial_graph = None
    records = to_link_records(
        lagged, field=gf, stability=stability, stability_threshold=stability_threshold,
        numerical_error=lagged.fit.numerical_error, spatial_graph=_spatial_graph,
        spatially_whitened=_whitened, temporally_whitened=temporal_whiten,
    )
    # §LDO-MARGIN-10 PIT-uniformity gate: flag edges whose endpoint count-exposure variable failed
    # the KS margin-calibration test (p<1e-3, strict to avoid chance failures) so certification
    # downgrades them — a distorted latent Z must not yield a certified edge.
    _miscal = {v for v, ksp in (getattr(gf, "margin_calibration", None) or {}).items() if ksp < 1e-3}
    if _miscal:
        records = [
            replace(r, warnings=r.warnings + ("margin_miscalibrated_descriptive_only",))
            if (r.source_var in _miscal or r.target_var in _miscal) else r
            for r in records
        ]

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

    # §III.8 conjuncts 3 & 4 (certgates): regularization-path agreement (an edge present at only a
    # single λ₁ is a threshold artefact) + latent-vs-lag separability (§IX.2: a directed lag whose
    # endpoints share a contemporaneous latent factor may be a phase-offset artefact). Annotate as
    # warnings; certify_link downgrades a promoted edge that fails either. Path agreement costs a
    # small λ-grid refit, so it is skipped when there is no directed/contemporaneous edge to gate.
    from pegasus.ldo.certgates import (
        apply_certification_gates,
        latent_vs_lag_confounds,
        regularization_path_agreement,
    )
    _latent_flags = latent_vs_lag_confounds(lagged)
    _path_agreement = None
    if any(r.edge_type in ("lagged_directed", "contemporaneous") for r in records):
        try:
            from pegasus.ldo.certgates import _edge_keys
            _path_agreement = regularization_path_agreement(
                gf, K=K, fit_kwargs=fit_kwargs, base_lambda1=lambda1, base_keys=_edge_keys(lagged))
        except Exception:
            _path_agreement = None
    records = apply_certification_gates(
        records, path_agreement=_path_agreement, latent_flags=_latent_flags)

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
    n_spatial_field_candidates = 0
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
        n_spatial_field_candidates = len(_cands)  # honest coverage: how many were eligible vs fitted
        for i, r in _cands[:max_spatial_fields]:
            try:
                sf = fit_spatial_varying_coefficient(gf, r.source_var, r.target_var, kappa=kappa)
                if sf is None:
                    continue
                _safe = f"{r.source_var}__{r.target_var}__lag{r.lag_k}".replace("/", "_")
                ref = write_spatial_field(sf, _sdir / f"{_safe}.spatial_field.parquet")
                # LDO-SEP-08(c): if the coupling's sign FLIPS across space (material β mass on both
                # sides of zero) surface it as a first-class warning — the single national partial
                # correlation is then a cancellation-prone average, and the edge is effect-modified,
                # not homogeneous. The β field sidecar carries the per-locality detail.
                _warns = r.warnings
                if sf.heterogeneity.get("sign_heterogeneous"):
                    _warns = _warns + ("spatial_sign_heterogeneous_coupling",)
                records[i] = replace(r, spatial_field_ref=ref, warnings=_warns)
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
            # Estimate the temporal factor's AR(1) φ from the data (mean lag-1 autocorrelation over
            # time, pooled across variables/space) instead of a hardcoded 0.5, so the separable
            # temporal precision reflects the panel's actual persistence.
            _phi = 0.5
            _za, _zb = gf.Z[:, :, :-1], gf.Z[:, :, 1:]
            _m = np.isfinite(_za) & np.isfinite(_zb)
            if int(_m.sum()) > 10:
                _a = _za[_m] - _za[_m].mean()
                _b = _zb[_m] - _zb[_m].mean()
                _den = float(np.sqrt((_a @ _a) * (_b @ _b)))
                if _den > 0:
                    _phi = float(np.clip((_a @ _b) / _den, -0.95, 0.95))
            _op = kronecker_from_ldo(lagged.lag0_precision, gf.space_ids, kappa=kappa, phi=_phi, tau=K + 1)
            _ld, _method = joint_logdet(_op, seed=seed)
            kronecker_report = {
                "joint_logdet": _ld, "space_logdet_method": _method, "temporal_ar1_phi": _phi,
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
        # §3.12.3/§II.3: number of variables whose reliability weight W was scaled by the Q-tensor
        # state (n_eff / denom_fragility / provenance_risk) — uncertain fields down-weighted, not exact.
        "n_state_weighted": len(field_weights) if field_weights else 0,
        # §V.1/§V.4 precision policy: the ADMM bulk dtype actually used, plus whether an
        # ill-conditioned covariance forced a float32→float64 escalation (self-correcting).
        "precision_policy": {
            "float32_bulk_requested": bool(f32),
            "work_dtype": str(getattr(lagged.fit, "work_dtype", "float64")),
            "cond_escalated": bool(getattr(lagged.fit, "cond_escalated", False)),
        },
        # Disease-axis effects (visible only when variable_meta/disease_graph were threaded):
        # the mechanical-overlap guard's re-typings and the disease-informed penalty.
        # §LDO-MARGIN-10 count-with-exposure margin calibration: the per-municipality baseline (Z is
        # deviation from the muni's OWN expected count, not the national rate) + the PIT-uniformity KS
        # gate. n_margin_miscalibrated variables failed KS (their edges downgraded to descriptive).
        "n_count_exposure_margins": len(getattr(gf, "margin_calibration", None) or {}),
        "n_margin_miscalibrated": len(_miscal),
        "margin_calibration_min_ks_p": (
            float(min((getattr(gf, "margin_calibration", None) or {}).values()))
            if getattr(gf, "margin_calibration", None) else None
        ),
        "disease_prior_applied": disease_penalty is not None,
        # §III.3/P2 adaptive disease shrinkage: whether the per-scale τ was data-estimated (vs the
        # flat γ), and the blocks whose borrow-strength is heavy (τ²>0.75) — their estimates lean on
        # hierarchical neighbours and must be read as shrunk, not independently measured.
        "disease_shrinkage_adaptive": disease_laplacian is not None and not disease_scale_precisions and raw_field is not None,
        "disease_heavily_shrunk_blocks": [
            {"scale": f["scale"], "variables": list(f["variables"]), "tau2": round(float(f["tau2"]), 3)}
            for f in disease_shrink_flags
        ],
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
        # §III.8 conjuncts 3 & 4: edges downgraded for regularization-path disagreement (single-λ
        # artefact) or a latent-vs-lag confound (§IX.2 phase-offset shared-wave artefact).
        "n_path_disagreement": sum(1 for r in records
                                   if any(w.startswith("low_regularization_path_agreement") for w in r.warnings)),
        "n_latent_lag_confound": sum(1 for r in records if "possible_latent_lag_confound" in r.warnings),
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
        # §V.6(3): exact-certifies-approximate on a real spatial slice — the randomized low-rank
        # readout is validated against an exact refit; certified=False flags a rejected approximation.
        "exact_certifies_approx": approx_certification,
        # §IX.3 temporal holdout: fraction of discovered edges that persist (same sign) on the
        # held-out tail year — the out-of-sample prong (None when the span is too short to split).
        "temporal_holdout": holdout_report,
        # §V.2: separable joint-precision log-det via the Kronecker-factored operator (None
        # when the variable precision is not SPD or the space factor is unavailable).
        "kronecker_joint": kronecker_report,
        # §III.3/§III.7: number of edges given a fitted spatial BYM varying-coefficient field vs
        # the number ELIGIBLE — so the top-N (max_spatial_fields) cap is an honest coverage bound,
        # not a silent truncation (edges beyond the cap carry spatial_field_ref=None by design).
        "n_spatial_fields": n_spatial_fields,
        "n_spatial_field_candidates": n_spatial_field_candidates,
        "spatial_field_cap": max_spatial_fields,
        # Theme-4 StARS: the λ₁ chosen by stability-of-regularization-selection + its
        # instability path (None when select_lambda is off — the default).
        "lambda_selection": lambda_selection,
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
    field_weights: dict | None = None,
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
        field_weights=field_weights,
    )
    coarse = coarsen_field_spatial(fine, level=coarse_level)
    ck = coarse_K if coarse_K is not None else max(1, min(K, 4))
    mr = run_multiresolution_ldo(coarse, fine, coarse_K=ck, fine_K=K, seed=seed, **ldo_kwargs)
    base = mr.fine.diagnostics if mr.fine is not None else mr.coarse.diagnostics
    diagnostics = dict(base)
    diagnostics["multiresolution"] = mr.diagnostics
    # §VIII.2(3): surface the typed coverage manifest (searched + explicitly-unsearched pairs +
    # the stated sparsity-of-truth assumption + the §VIII.2(2) random-audit false-negative rate)
    # so the honesty record reaches the run bundle, not just the coarse-pass diagnostics.
    if mr.coverage_manifest is not None:
        cm = mr.coverage_manifest
        diagnostics["coverage_manifest_mr"] = {
            "searched": cm.searched,
            "unsearched": cm.unsearched,
            "sparsity_of_truth_assumption": cm.sparsity_of_truth_assumption,
            "random_audit_false_negative_rate": cm.random_audit_false_negative_rate,
        }
    return LDORun(link_records=mr.link_records, variables=fine.variables, diagnostics=diagnostics)


__all__ = ["LDORun", "run_ldo"]
