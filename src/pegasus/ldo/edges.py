"""LDO edge readout + stability selection (MSD-II §II.6.3, MII-LDO-05).

Turn a fitted ``{S, L}`` (with lag structure) into typed ``LinkRecord``s, and
control multiplicity by **stability selection**: refit on lattice subsamples and
keep only edges recurring above a frequency threshold (finite-sample edge-set
control without ``p²`` separate tests). Edges below the stability threshold are
marked ``descriptive`` rather than promoted; low-power (``n_eff < 100``) edges are
descriptive only.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from pegasus.ldo.lags import LaggedFit, fit_lagged_links
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.records import LinkRecord

_MIN_N_EFF = 100


def _edge_key(source: str, target: str, lag: int) -> tuple[str, str, int]:
    return (source, target, lag)


def _subsample_edges(
    field: GaussianField, idx: np.ndarray, *, K: int, fit_kwargs: dict,
    t_idx: np.ndarray | None = None,
) -> set[tuple[str, str, int]] | None:
    """Fit one subsample and return its selected edge keys (or None on failure).

    ``idx`` selects the SPATIAL subsample; ``t_idx`` (optional) selects a contiguous TIME window
    so an edge that survives only at one time span is filtered (§III.7/§IX.3 lattice-subsampling
    across Place AND Time, not space alone)."""
    tsel = slice(None) if t_idx is None else t_idx
    time_ids = field.time_ids if t_idx is None else tuple(field.time_ids[t] for t in t_idx)
    sub = GaussianField(
        variables=field.variables,
        space_ids=tuple(field.space_ids[i] for i in idx),
        time_ids=time_ids,
        Z=field.Z[:, idx, :][:, :, tsel] if t_idx is not None else field.Z[:, idx, :],
        W=field.W[:, idx, :][:, :, tsel] if t_idx is not None else field.W[:, idx, :],
        resolution=field.resolution,
    )
    try:
        res = fit_lagged_links(sub, K=min(K, sub.Z.shape[2] - 2), **fit_kwargs)
    except Exception:
        return None
    seen: set[tuple[str, str, int]] = set()
    for lk in res.lagged_links:
        seen.add(_edge_key(lk.source, lk.target, lk.peak_lag))
    for src, tgt, _ in res.contemporaneous:
        seen.add(_edge_key(src, tgt, 0))
    return seen


def stability_select(
    field: GaussianField,
    *,
    K: int = 8,
    n_subsamples: int = 20,
    subsample_frac: float = 0.7,
    lag_tolerance: int = 1,
    seed: int = 0,
    max_workers: int | None = None,
    perturb_time: bool | str = "auto",
    **fit_kwargs,
) -> dict[tuple[str, str, int], float]:
    """Return per-edge selection frequency over lattice subsamples (Place × Time).

    Each subsample perturbs the SPATIAL lattice and — when ``perturb_time`` is on ("auto" enables
    it once the span is long enough to leave ``> K`` points, ``T ≥ K+4``) — also a contiguous TIME
    window, so an edge that survives only at one spatial OR one temporal span is filtered
    (§III.7/§IX.3 "edges must survive subsampling", across Place AND Time, not space alone).

    The subsample refits are independent and dominate the LDO cost, so they run
    concurrently on a thread pool (numpy's LAPACK eigensolves release the GIL and
    the field arrays are shared read-only — no per-worker serialization). Draw all
    subsample indices up front so the result is deterministic regardless of
    completion order.
    """
    rng = np.random.default_rng(seed)
    p, S, T = field.shape
    n_keep = max(2, int(round(subsample_frac * S)))
    index_sets = [np.sort(rng.choice(S, size=min(n_keep, S), replace=False)) for _ in range(n_subsamples)]
    use_time = (T >= K + 4) if perturb_time == "auto" else bool(perturb_time)
    if use_time:
        # §LDO-TIME-03 sharpening: a single ≥0.6·T window makes every subsample cover almost the
        # whole span, so a trend-driven spurious edge is present (and thus "stable") in all of them —
        # contiguous subsampling CONFIRMS the nonsense correlation. Allow shorter windows (≥0.4·T,
        # still ≥K+4 for lag formation) at dispersed starts so early/mid/late epochs are sampled
        # distinctly; a full-span-trend-only edge then fails to appear in the short off-epoch windows.
        t_min = max(K + 4, int(round(0.4 * T)))
        time_sets: list[np.ndarray | None] = []
        for _ in range(n_subsamples):
            L = int(rng.integers(t_min, T + 1))
            start = int(rng.integers(0, T - L + 1))
            time_sets.append(np.arange(start, start + L))
    else:
        time_sets = [None] * n_subsamples

    if max_workers is None:
        max_workers = max(1, min(n_subsamples, (os.cpu_count() or 2) - 1))
    if max_workers <= 1:
        results = [_subsample_edges(field, idx, K=K, fit_kwargs=fit_kwargs, t_idx=t)
                   for idx, t in zip(index_sets, time_sets)]
    else:
        # Pin BLAS to one thread per worker so N eigensolves use N cores rather than
        # oversubscribing (each LAPACK eigh would otherwise grab every core).
        try:
            from threadpoolctl import threadpool_limits
            limiter = threadpool_limits(limits=1, user_api="blas")
        except Exception:
            limiter = None
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(pool.map(
                    lambda it: _subsample_edges(field, it[0], K=K, fit_kwargs=fit_kwargs, t_idx=it[1]),
                    list(zip(index_sets, time_sets))))
        finally:
            if limiter is not None:
                limiter.unregister()

    counts: dict[tuple[str, str, int], int] = {}
    runs = 0
    for seen in results:
        if seen is None:
            continue
        runs += 1
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    if runs == 0:
        return {}
    # merge lag-adjacent keys within tolerance onto their strongest bucket
    freq: dict[tuple[str, str, int], float] = {k: v / runs for k, v in counts.items()}
    if lag_tolerance > 0:
        merged: dict[tuple[str, str, int], float] = {}
        for (src, tgt, lag), f in sorted(freq.items(), key=lambda kv: kv[1], reverse=True):
            hit = None
            for (s2, t2, l2) in merged:
                if s2 == src and t2 == tgt and abs(l2 - lag) <= lag_tolerance:
                    hit = (s2, t2, l2)
                    break
            if hit is None:
                merged[(src, tgt, lag)] = f
            else:
                merged[hit] = min(1.0, merged[hit] + f)
        freq = merged
    return freq


def _global_n_eff(field: GaussianField) -> int:
    return int(np.isfinite(field.Z).any(axis=0).sum())


def _node_summary(zvar: np.ndarray, wvar: np.ndarray, both: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-space reliability-weighted mean of one endpoint over joint-observed cells,
    plus per-space summed weight. ``zvar/wvar/both`` are (S, T)."""
    w = np.where(both, wvar, 0.0)
    wsum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, (w * np.nan_to_num(zvar)).sum(axis=1) / wsum, np.nan)
    return mean, wsum


def _spatial_deflation_map(field: GaussianField, graph, provenance) -> dict[str, float]:
    """Per-variable spatial deflation ratio effective_n/n ∈ (0,1] over each variable's marginal
    reliability-weighted spatial support — computed ONCE per variable (not per edge), matrix-free."""
    from pegasus.geo.spatial import effective_n as _sen
    out: dict[str, float] = {}
    for i, v in enumerate(field.variables):
        both = np.isfinite(field.Z[i])
        mean, _ = _node_summary(field.Z[i], field.W[i], both)
        vals = {sid: mean[s] for s, sid in enumerate(field.space_ids) if np.isfinite(mean[s])}
        n_marg = len(vals)
        if n_marg < 3:
            out[v] = 1.0
            continue
        try:
            ne = _sen(vals, graph, variable_provenance=provenance)
        except Exception:
            ne = None
        out[v] = float(ne / n_marg) if (ne is not None and n_marg > 0) else 1.0
    return out


def _edge_n_eff(
    field: GaussianField, source: str, target: str,
    defl: dict[str, float] | None, phi: dict[str, float] | None = None,
) -> float:
    """Effective sample size for one edge: reliability-weighted joint-observed cell count,
    scaled by the worse endpoint's spatial deflation and — when time is un-whitened — the pair's
    temporal-autocorrelation design effect (§LDO-TIME-03/LDO-NEFF-15). Both shrink n_eff for
    positively-autocorrelated endpoints so the Fisher-z SE is not computed on inflated iid counts."""
    idx = {v: i for i, v in enumerate(field.variables)}
    i, j = idx.get(source), idx.get(target)
    if i is None or j is None:
        return float(_global_n_eff(field))
    both = np.isfinite(field.Z[i]) & np.isfinite(field.Z[j])
    rel = float(np.where(both, np.minimum(field.W[i], field.W[j]), 0.0).sum())
    if rel <= 0:
        return 0.0
    if defl:
        rel *= min(defl.get(source, 1.0), defl.get(target, 1.0))
    if phi:
        # Bartlett's cross-correlation variance for two AR(1) series: Var(r_ab) ≈
        # (1/n)·Σ_k φ_a^|k|φ_b^|k| = (1/n)·(1+φ_aφ_b)/(1−φ_aφ_b), so the honest effective-n is
        # n·(1−φ_aφ_b)/(1+φ_aφ_b). (This is the correct pairwise estimand — the single-series
        # (1−φ)/(1+φ) is the sample-mean design effect, not the correlation's.) Capped at 1 so
        # anti-persistence (φ_aφ_b<0) is never CREDITED as extra independent samples — conservative.
        prod = phi.get(source, 0.0) * phi.get(target, 0.0)
        fac = (1.0 - prod) / (1.0 + prod) if (1.0 + prod) > 1e-9 else 1.0
        rel *= min(1.0, max(fac, 1e-6))
    return float(rel)


def _temporal_phi_map(field: GaussianField) -> dict[str, float]:
    """Per-variable within-unit AR(1) φ̂ for the serial-correlation effective-n deflation —
    reuses ``temporal._ar1_phi`` (each spatial unit demeaned by its own temporal mean, so
    between-unit level differences can't masquerade as persistence). Computed ONCE per variable."""
    from pegasus.ldo.temporal import _ar1_phi
    return {v: _ar1_phi(field.Z[i]) for i, v in enumerate(field.variables)}


def to_link_records(
    lagged: LaggedFit,
    *,
    field: GaussianField | None = None,
    stability: dict[tuple[str, str, int], float] | None = None,
    stability_threshold: float = 0.5,
    null_strategy: str | None = None,
    fdr_method: str | None = None,
    numerical_error: float = 0.0,
    spatial_graph=None,
    variable_provenance=None,
    per_edge_n_eff: bool = True,
    spatially_whitened: bool = False,
    temporally_whitened: bool = False,
) -> list[LinkRecord]:
    """Assemble typed LinkRecords from a lagged fit, gated by stability + power.

    ``per_edge_n_eff`` (default) sizes each edge's Fisher-z SE and low-power gate from a
    per-edge effective-n over its two endpoints' joint-observed cells — spatially corrected via
    ``spatial_graph`` when supplied, reliability-weighted by ``field.W`` otherwise. Off falls back
    to the single global observed-cell count (prior behaviour).

    ``spatially_whitened`` (Theme-6 double-correction guard): when the fit was GMRF-whitened across
    space the whitened observations are already ~spatially independent, so the effective-n must NOT
    also be Moran-deflated (that would count the spatial dependence twice and over-inflate the SE).
    In that case only the reliability weighting is applied; the ``spatial_graph`` deflation is used
    only for a non-whitened fit.

    ``temporally_whitened`` (§LDO-TIME-03 double-correction guard, dual of the spatial one): the
    overlapping lag windows / serial autocorrelation make the S·T_eff cells non-iid, so an
    un-whitened fit's effective-n is deflated by each pair's AR(1) Bartlett design effect. When the
    fit WAS temporally whitened the residuals are already ~serially independent, so that deflation
    is skipped (else the serial dependence would be counted twice)."""
    stability = stability or {}
    import math as _math
    global_n = _global_n_eff(field) if field is not None else None
    # Each edge is a PARTIAL correlation off the precision matrix, conditioning on the other
    # q−2 kept features (q = precision dimension). The Fisher-z dof is n_eff − (q−2) − 3 = n_eff −
    # q − 1, not n_eff − 3; using the marginal dof understates the SE (anticonservative FDR).
    _q = int(len(lagged.fit.S)) if getattr(lagged, "fit", None) is not None and getattr(lagged.fit, "S", None) is not None else 0
    k_cond = max(_q - 2, 0)

    # §III.4.2 identifiability: when the sparse+low-rank split is poorly identified (S mass sits
    # inside L's span, CPW incoherence low) every edge read off it is suspect — flag them so a
    # reader does not treat a possibly-arbitrary direct/latent edge as certified structure.
    _split_suspect = getattr(getattr(lagged, "fit", None), "well_identified", True) is False

    def _power_warnings(low: bool, underdet: bool) -> tuple[str, ...]:
        w: list[str] = []
        if underdet:
            w.append("partial_corr_underdetermined")
        elif low:
            w.append("low_n_eff_descriptive_only")
        if _split_suspect:
            w.append("poorly_identified_split")
        return tuple(w)

    def _uncertainty(n: float | None) -> tuple[float | None, bool, float | None, bool]:
        if n is None:
            return None, False, None, False
        dof = n - k_cond - 3.0
        if dof < 1.0:  # partial correlation not identifiable at this effective-n → refuse a finite SE
            return None, True, float(n), True
        se = 1.0 / _math.sqrt(dof)
        # §III.7/§V.6(2): Fisher-z SE combined in quadrature with the fit's numerical error.
        unc = float(_math.hypot(se, float(numerical_error)))
        return unc, n < _MIN_N_EFF, float(n), False

    use_edge = per_edge_n_eff and field is not None
    defl = (
        _spatial_deflation_map(field, spatial_graph, variable_provenance)
        if use_edge and spatial_graph is not None and not spatially_whitened else None
    )
    # §LDO-TIME-03 / LDO-NEFF-15: un-whitened time ⇒ deflate each edge's effective-n by the pair's
    # AR(1) serial-correlation design effect so a trend-driven partial correlation gets an honestly
    # wide SE (and fails certification) instead of spuriously-tight iid significance.
    phi_map = _temporal_phi_map(field) if (use_edge and not temporally_whitened) else None

    def _edge_stats(source: str, target: str) -> tuple[float | None, bool, float | None, bool]:
        if use_edge:
            return _uncertainty(_edge_n_eff(field, source, target, defl, phi_map))
        return _uncertainty(global_n)

    records: list[LinkRecord] = []
    for lk in lagged.lagged_links:
        key = _edge_key(lk.source, lk.target, lk.peak_lag)
        stab = stability.get(key)
        if stab is None:  # tolerate ±1 lag bucket
            for (s2, t2, l2), v in stability.items():
                if s2 == lk.source and t2 == lk.target and abs(l2 - lk.peak_lag) <= 1:
                    stab = v
                    break
        edge_uncertainty, low_power, edge_n, underdet = _edge_stats(lk.source, lk.target)
        certified = (not low_power) and (stab is None or stab >= stability_threshold)
        records.append(
            LinkRecord(
                source_var=lk.source,
                target_var=lk.target,
                edge_type="lagged_directed",
                lag_k=lk.peak_lag,
                weight=lk.peak_partial_correlation,
                partial_correlation=lk.peak_partial_correlation,
                response_curve_ref=",".join(f"{r:.4f}" for r in lk.response_curve),
                stability=stab,
                uncertainty=edge_uncertainty,
                n_eff=edge_n,
                n_conditioning=k_cond,
                certification_status="selected" if certified else "descriptive",
                null_strategy=null_strategy,
                fdr_method=fdr_method,
                warnings=_power_warnings(low_power, underdet),
            )
        )
    for source, target, pcorr in lagged.contemporaneous:
        edge_uncertainty, low_power, edge_n, underdet = _edge_stats(source, target)
        records.append(
            LinkRecord(
                source_var=source,
                target_var=target,
                edge_type="contemporaneous",
                lag_k=0,
                weight=pcorr,
                partial_correlation=pcorr,
                stability=stability.get(_edge_key(source, target, 0)),
                uncertainty=edge_uncertainty,
                n_eff=edge_n,
                n_conditioning=k_cond,
                certification_status="descriptive" if low_power else "selected",
                warnings=_power_warnings(low_power, underdet),
            )
        )
    for source, target, loading in lagged.latent_shared:
        edge_uncertainty, low_power, edge_n, underdet = _edge_stats(source, target)
        records.append(
            LinkRecord(
                source_var=source,
                target_var=target,
                edge_type="latent_shared",
                weight=loading,
                confounding_factor_refs=("ldo_low_rank_factor",),
                uncertainty=edge_uncertainty,
                n_eff=edge_n,
                n_conditioning=k_cond,
                certification_status="descriptive" if low_power else "selected",
                warnings=_power_warnings(low_power, underdet),
            )
        )
    return records


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def type_mechanical_overlap(
    records: list[LinkRecord],
    variable_code_sets: dict[str, frozenset[str]],
    *,
    threshold: float = 0.5,
) -> list[LinkRecord]:
    """Re-type edges between concept-variables that share underlying codes (§II.6/§5.3).

    Two disease-concept variables built on overlapping code sets are *mechanically*
    correlated — they count overlapping events — so such an edge is NOT an
    epidemiological discovery. When the Jaccard overlap of their code sets exceeds
    ``threshold`` the edge is typed ``mechanical_overlap`` and demoted to descriptive;
    every edge whose variables both carry code sets is annotated with ``overlap_jaccard``.
    """
    # A shared-code artefact can surface as a direct edge OR as a shared latent factor
    # (the low-rank layer absorbs the shared count); both must be re-typed, never a discovery.
    _overlappable = {"contemporaneous", "lagged_directed", "latent_shared"}
    out: list[LinkRecord] = []
    for r in records:
        cs_s = variable_code_sets.get(r.source_var)
        cs_t = variable_code_sets.get(r.target_var)
        if cs_s is None or cs_t is None or r.edge_type not in _overlappable:
            out.append(r)
            continue
        j = _jaccard(cs_s, cs_t)
        if j >= threshold:
            out.append(replace(
                r, edge_type="mechanical_overlap", overlap_jaccard=j,
                certification_status="descriptive",
                warnings=r.warnings + ("mechanical_overlap_shared_codes",),
            ))
        else:
            out.append(replace(r, overlap_jaccard=j))
    return out


_PROJECTION_SEVERITY = {
    "exact": 0, "source_system_specific": 1, "parent_projection": 2,
    "approximate": 3, "unmappable": 4,
}


def annotate_disease_provenance(
    records: list[LinkRecord], variable_meta: dict[str, dict]
) -> list[LinkRecord]:
    """Stamp each edge with disease-axis provenance from its variables' metadata (§III.7).

    ``code_system``/``topology_role`` are set when both endpoints agree (else left None,
    an honest "mixed"); ``projection_status`` is the worst of the two (anti-false-precision:
    an ``approximate`` CCSR endpoint forces the edge to ``approximate``).
    """
    out: list[LinkRecord] = []
    for r in records:
        ms = variable_meta.get(r.source_var) or {}
        mt = variable_meta.get(r.target_var) or {}
        if not ms and not mt:
            out.append(r)
            continue

        def _agree(key: str) -> str | None:
            a, b = ms.get(key), mt.get(key)
            return a if a is not None and a == b else (a or b if not (a and b) else None)

        statuses = [s for s in (ms.get("projection_status"), mt.get("projection_status")) if s]
        projection = max(statuses, key=lambda s: _PROJECTION_SEVERITY.get(s, 0)) if statuses else None
        out.append(replace(
            r, code_system=_agree("code_system"), topology_role=_agree("topology_role"),
            projection_status=projection,
        ))
    return out


__all__ = ["stability_select", "to_link_records", "type_mechanical_overlap", "annotate_disease_provenance"]
