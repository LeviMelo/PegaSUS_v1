"""The two §III.8 certification conjuncts the original audit missed (MSD-III §III.8, §IX.2).

The §III.8 gate names FOUR conjuncts: holdout stability, propagated uncertainty, **regularization-
path agreement**, and **latent-vs-lag separability**. The first two are wired (certify.py); these
two were absent. Both are computed here and folded into the gate as edge warnings that
:func:`pegasus.ldo.certify.certify_link` downgrades on.

* **Regularization-path agreement** — an edge that appears only at a single graphical-lasso
  ``λ₁`` operating point is fragile (a threshold artefact). We refit along a small ``λ₁`` grid and
  keep the fraction of the grid on which each edge is selected; an edge below the agreement floor
  is not certified.

* **Latent-vs-lag separability (§IX.2 certification power)** — a directed lag-``k`` edge whose two
  endpoints ALSO share a contemporaneous latent factor may be an artefact of two variables riding
  the same wave at a phase offset, not a genuine lagged mechanism. We flag such edges (the CPW
  low-rank component already recovers the shared factor as a ``latent_shared`` pair).
"""

from __future__ import annotations

from dataclasses import replace

from pegasus.ldo.records import LinkRecord


def _edge_keys(lagged) -> set[tuple[frozenset, int]]:
    keys: set[tuple[frozenset, int]] = set()
    for lk in lagged.lagged_links:
        keys.add((frozenset((lk.source, lk.target)), lk.peak_lag))
    for s, t, _ in lagged.contemporaneous:
        keys.add((frozenset((s, t)), 0))
    return keys


def regularization_path_agreement(
    field, *, K: int, fit_kwargs: dict, base_lambda1: float,
    grid: tuple[float, ...] = (0.5, 2.0), base_keys: set | None = None,
) -> dict[tuple[frozenset, int], float]:
    """Fraction of the ``λ₁`` grid (base × 1 plus each multiplier) on which each edge is selected.

    Refits :func:`pegasus.ldo.lags.fit_lagged_links` at ``base_lambda1 × g`` for each ``g`` and
    unions the selected edge keys with the base fit's (``base_keys``). An edge present across the
    grid scores ~1.0; a single-``λ`` artefact scores ``1/(1+len(grid))``.
    """
    from pegasus.ldo.lags import fit_lagged_links

    per_lambda: dict[float, set] = {1.0: set(base_keys or set())}
    for g in grid:
        fk = dict(fit_kwargs)
        fk["lambda1"] = base_lambda1 * g
        per_lambda[g] = _edge_keys(fit_lagged_links(field, K=K, **fk))
    grids = [1.0, *grid]
    all_keys: set = set().union(*per_lambda.values())
    return {
        key: sum(1 for g in grids if key in per_lambda.get(g, set())) / len(grids)
        for key in all_keys
    }


def latent_vs_lag_confounds(lagged, *, loading_floor: float = 0.0) -> set[tuple[str, str, int]]:
    """Directed lagged edges whose endpoints ALSO form a ``latent_shared`` (shared-factor) pair.

    Such an edge may be two variables riding the same contemporaneous wave at a phase offset
    rather than a genuine lagged mechanism (§IX.2). Returns the flagged ``(source, target, lag)``
    keys."""
    latent_pairs = {
        frozenset((a, b)) for a, b, load in lagged.latent_shared if abs(load) >= loading_floor
    }
    return {
        (lk.source, lk.target, lk.peak_lag)
        for lk in lagged.lagged_links
        if frozenset((lk.source, lk.target)) in latent_pairs
    }


def apply_certification_gates(
    records: list[LinkRecord], *,
    path_agreement: dict[tuple[frozenset, int], float] | None = None,
    latent_flags: set[tuple[str, str, int]] | None = None,
    min_path_agreement: float = 0.66,
) -> list[LinkRecord]:
    """Annotate records with the two §III.8 conjunct warnings (certify_link downgrades on them)."""
    path_agreement = path_agreement or {}
    latent_flags = latent_flags or set()
    out: list[LinkRecord] = []
    for r in records:
        warns = r.warnings
        if r.edge_type in ("lagged_directed", "contemporaneous"):
            key = (frozenset((r.source_var, r.target_var)), r.lag_k)
            agree = path_agreement.get(key)
            if agree is not None and agree < min_path_agreement:
                warns = warns + (f"low_regularization_path_agreement:{agree:.2f}",)
        if r.edge_type == "lagged_directed" and (r.source_var, r.target_var, r.lag_k) in latent_flags:
            warns = warns + ("possible_latent_lag_confound",)
        out.append(replace(r, warnings=warns) if warns != r.warnings else r)
    return out


__all__ = [
    "regularization_path_agreement",
    "latent_vs_lag_confounds",
    "apply_certification_gates",
]
