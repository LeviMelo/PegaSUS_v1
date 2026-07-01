"""LDO edge readout + stability selection (MSD-II §II.6.3, MII-LDO-05).

Turn a fitted ``{S, L}`` (with lag structure) into typed ``LinkRecord``s, and
control multiplicity by **stability selection**: refit on lattice subsamples and
keep only edges recurring above a frequency threshold (finite-sample edge-set
control without ``p²`` separate tests). Edges below the stability threshold are
marked ``descriptive`` rather than promoted; low-power (``n_eff < 100``) edges are
descriptive only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pegasus.pirs.ldo.lags import LaggedFit, fit_lagged_links
from pegasus.pirs.ldo.margins import GaussianField
from pegasus.pirs.ldo.records import LinkRecord

_MIN_N_EFF = 100


def _edge_key(source: str, target: str, lag: int) -> tuple[str, str, int]:
    return (source, target, lag)


def stability_select(
    field: GaussianField,
    *,
    K: int = 8,
    n_subsamples: int = 20,
    subsample_frac: float = 0.7,
    lag_tolerance: int = 1,
    seed: int = 0,
    **fit_kwargs,
) -> dict[tuple[str, str, int], float]:
    """Return per-edge selection frequency over spatial subsamples of the lattice."""
    rng = np.random.default_rng(seed)
    p, S, T = field.shape
    counts: dict[tuple[str, str, int], int] = {}
    n_keep = max(2, int(round(subsample_frac * S)))
    runs = 0
    for _ in range(n_subsamples):
        idx = np.sort(rng.choice(S, size=min(n_keep, S), replace=False))
        sub = GaussianField(
            variables=field.variables,
            space_ids=tuple(field.space_ids[i] for i in idx),
            time_ids=field.time_ids,
            Z=field.Z[:, idx, :],
            W=field.W[:, idx, :],
            resolution=field.resolution,
        )
        try:
            res = fit_lagged_links(sub, K=K, **fit_kwargs)
        except Exception:
            continue
        runs += 1
        seen: set[tuple[str, str, int]] = set()
        for lk in res.lagged_links:
            # bucket peak lags within tolerance so a jittering peak still counts
            key = _edge_key(lk.source, lk.target, lk.peak_lag)
            seen.add(key)
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


def _n_eff(field: GaussianField) -> int:
    return int(np.isfinite(field.Z).any(axis=0).sum())


def to_link_records(
    lagged: LaggedFit,
    *,
    field: GaussianField | None = None,
    stability: dict[tuple[str, str, int], float] | None = None,
    stability_threshold: float = 0.5,
    null_strategy: str | None = None,
    fdr_method: str | None = None,
) -> list[LinkRecord]:
    """Assemble typed LinkRecords from a lagged fit, gated by stability + power."""
    stability = stability or {}
    n_eff = _n_eff(field) if field is not None else None
    low_power = n_eff is not None and n_eff < _MIN_N_EFF

    records: list[LinkRecord] = []
    for lk in lagged.lagged_links:
        key = _edge_key(lk.source, lk.target, lk.peak_lag)
        stab = stability.get(key)
        if stab is None:  # tolerate ±1 lag bucket
            for (s2, t2, l2), v in stability.items():
                if s2 == lk.source and t2 == lk.target and abs(l2 - lk.peak_lag) <= 1:
                    stab = v
                    break
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
                certification_status="selected" if certified else "descriptive",
                null_strategy=null_strategy,
                fdr_method=fdr_method,
                warnings=("low_n_eff_descriptive_only",) if low_power else (),
            )
        )
    for source, target, loading in lagged.latent_shared:
        records.append(
            LinkRecord(
                source_var=source,
                target_var=target,
                edge_type="latent_shared",
                weight=loading,
                confounding_factor_refs=("ldo_low_rank_factor",),
                certification_status="descriptive" if low_power else "selected",
                warnings=("low_n_eff_descriptive_only",) if low_power else (),
            )
        )
    return records


__all__ = ["stability_select", "to_link_records"]
