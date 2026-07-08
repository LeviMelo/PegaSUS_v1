"""Standing validation prongs: exact-certifies-approximate + temporal holdout (MSD-III §IX.3, §V.6).

Two of the Part IX certification prongs that must actually RUN on a real slice of the data, not
merely exist as callable comparators:

* :func:`certify_approximation_on_slice` — §V.6(3). The national LDO uses approximations
  (randomized SVD low-rank readout, matrix-free whitening). On a bounded, real spatial slice we
  run the SAME pipeline BOTH approximate and EXACT and compare the overlapping edges through
  :func:`pegasus.ldo.exact_certify.certify_exact_vs_approx`. Agreement within the propagated band
  certifies the approximation *method* for the national run; disagreement rejects it LOUDLY.

* :func:`temporal_holdout` — §IX.3. Fit through year ``T``, verify the edge on ``T+1``: a real
  link persists / predicts; a fluke evaporates. (Built in the O5 remediation.)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pegasus.ldo.exact_certify import certify_exact_vs_approx
from pegasus.ldo.margins import GaussianField


def _spatial_slice(field: GaussianField, max_slice_S: int) -> GaussianField:
    """The ``max_slice_S`` most-observed localities — a bounded slice where an EXACT eigh fit is
    affordable, drawn from the real data so the certification is representative."""
    coverage = np.isfinite(field.Z).sum(axis=(0, 2))       # (S,) finite cells per locality
    keep = np.argsort(-coverage)[:max_slice_S]
    keep = np.sort(keep)
    return GaussianField(
        variables=field.variables,
        space_ids=tuple(field.space_ids[i] for i in keep),
        time_ids=field.time_ids,
        Z=field.Z[:, keep, :],
        W=field.W[:, keep, :],
        resolution=field.resolution,
    )


def certify_approximation_on_slice(
    field: GaussianField,
    *,
    K: int,
    fit_kwargs: dict | None = None,
    max_slice_S: int = 700,
    tolerance_sigma: float = 3.0,
    strict: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """§V.6(3) exact-certifies-approximate on a real spatial slice.

    Runs :func:`pegasus.ldo.lags.fit_lagged_links` twice on the slice — once with the randomized
    low-rank readout (``randomized_factors=True``, the national approximation) and once EXACT
    (``randomized_factors=False``) — reads both to LinkRecords, and compares overlapping edges.
    Returns the comparator report augmented with the slice size; ``strict=True`` re-raises
    :class:`~pegasus.ldo.exact_certify.ApproximationRejectedError` on disagreement. Skipped
    (``ran=False``) when the field is too small for the approximation to differ from exact.
    """
    from pegasus.ldo.edges import to_link_records
    from pegasus.ldo.lags import fit_lagged_links

    fit_kwargs = dict(fit_kwargs or {})
    fit_kwargs.pop("randomized_factors", None)
    sl = _spatial_slice(field, max_slice_S) if field.Z.shape[1] > max_slice_S else field
    if sl.Z.shape[1] < 3:
        return {"ran": False, "reason": "slice_too_small", "certified": True, "n_overlap": 0}

    approx = fit_lagged_links(sl, K=K, randomized_factors=True, **fit_kwargs)
    exact = fit_lagged_links(sl, K=K, randomized_factors=False, **fit_kwargs)
    approx_records = to_link_records(approx, field=sl, numerical_error=approx.fit.numerical_error)
    exact_records = to_link_records(exact, field=sl, numerical_error=exact.fit.numerical_error)
    report = certify_exact_vs_approx(
        approx_records, exact_records, tolerance_sigma=tolerance_sigma, strict=strict)
    report["ran"] = True
    report["slice_S"] = int(sl.Z.shape[1])
    return report


def _time_slice(field: GaussianField, t_lo: int, t_hi: int) -> GaussianField:
    idx = list(range(t_lo, t_hi))
    return GaussianField(
        variables=field.variables, space_ids=field.space_ids,
        time_ids=tuple(field.time_ids[t] for t in idx),
        Z=field.Z[:, :, idx], W=field.W[:, :, idx], resolution=field.resolution,
    )


def _edge_signset(records) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for r in records:
        if r.edge_type in ("contemporaneous", "lagged_directed"):
            out[(frozenset((r.source_var, r.target_var)), r.lag_k)] = float(np.sign(r.weight or 0.0))
    return out


def temporal_holdout(
    field: GaussianField,
    *,
    K: int = 1,
    fit_kwargs: dict | None = None,
    holdout_years: int = 1,
    min_train_T: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """§IX.3 temporal holdout: fit through year ``T``, verify on the held-out tail.

    Fits the LDO on the training window (all but the last ``holdout_years``) and on the held-out
    window, then measures the **persistence rate** — the fraction of training-selected directed /
    contemporaneous edges that recur with the SAME sign in the holdout window. A real link
    persists / predicts; a fluke evaporates. Returns the rate + counts; skipped (``ran=False``)
    when the time span is too short to split. Both windows must clear ``K``.
    """
    from pegasus.ldo.edges import to_link_records
    from pegasus.ldo.lags import fit_lagged_links

    fit_kwargs = dict(fit_kwargs or {})
    p, S, T = field.Z.shape
    train_T = T - holdout_years
    test_T = T - train_T
    if train_T < max(min_train_T, 2) or holdout_years < 1 or test_T < 2:
        return {"ran": False, "reason": "time_span_too_short", "persistence_rate": None}
    # Fit BOTH windows at the same affordable lag order so their edge keys are comparable
    # (a directed edge the shorter window cannot even represent must not count as "not persisted").
    k_eff = max(0, min(K, train_T - 2, test_T - 2))

    def _edges(sl):
        lag = fit_lagged_links(sl, K=k_eff, **fit_kwargs)
        return _edge_signset(to_link_records(lag, field=sl))

    train = _edges(_time_slice(field, 0, train_T))
    test = _edges(_time_slice(field, train_T, T))
    if not train:
        return {"ran": True, "persistence_rate": None, "n_train_edges": 0,
                "n_persisted": 0, "reason": "no_train_edges"}
    persisted = sum(1 for k, s in train.items() if test.get(k) == s and s != 0.0)
    n = len(train)
    # §LDO-CERT-HOLDOUT-NULL-10: sign-persistence has a chance baseline — a fluke edge's sign recurs in
    # the holdout with probability ~0.5. Report the persistence AGAINST that null via a one-sided
    # binomial test (is the observed count beyond chance sign-matching?), not the bare rate, which is
    # otherwise uninterpretable (0.6 could be pure chance).
    from scipy.stats import binom
    p_value = float(binom.sf(persisted - 1, n, 0.5)) if n > 0 else None
    return {
        "ran": True,
        "n_train_edges": n,
        "n_persisted": persisted,
        "persistence_rate": float(persisted) / float(n),
        "persistence_null_rate": 0.5,
        "persistence_pvalue": p_value,
        "holdout_years": holdout_years,
    }


__all__ = ["certify_approximation_on_slice", "temporal_holdout"]
