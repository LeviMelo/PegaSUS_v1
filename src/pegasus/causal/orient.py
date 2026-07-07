"""Rung-1 causal orientation (MSD-III §IV, CAUSAL-01).

Turn the LDO's *undirected* contemporaneous edges into *directed* causal hypotheses
where — and only where — the direction is machine-identifiable:

- **Non-Gaussian orientation (additive-noise / LiNGAM).** For a pair with a linear
  relationship and non-Gaussian disturbances, the direction is identifiable: if X→Y
  then the residual of Y regressed on X is independent of X, but the reverse residual
  is *not* independent of Y. The direction whose residual is more independent of its
  putative cause is the causal one. Epidemiological counts are non-Gaussian, so this
  extracts real orientation the LDO does not natively provide. Where the data is ~Gaussian
  (direction unidentifiable) the edge is left undirected, tagged honestly.
- **Collider (v-structure) detection.** For a triple A–C–B where A,B are each adjacent
  to C but not to each other, if A⊥B but A⊥̸B∣C then the structure is A→C←B.

Lagged edges are already directed by time (past→present). Every oriented edge carries
its licensing check in ``warnings``; nothing is oriented beyond what is identifiable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.stats import normaltest

from pegasus.ldo.records import LinkRecord

_EPS = 1e-9


def _standardize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + _EPS)


def is_non_gaussian(a: np.ndarray, *, alpha: float = 0.01) -> bool:
    """The identifiability licence for LiNGAM: does ``a`` reject normality?

    Uses the D'Agostino–Pearson test (skew + kurtosis). Conservative by design
    (``alpha=0.01``): we orient only when a variable is *clearly* non-Gaussian, so a
    genuinely Gaussian pair — where the direction is unidentifiable — is left undirected
    rather than spuriously oriented on sampling noise.
    """
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 20:
        return False
    try:
        _, p = normaltest(a)
    except Exception:
        return False
    return bool(p < alpha)


def _residual(target: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    """OLS residual of ``target`` on ``regressor`` (both standardized)."""
    t, r = _standardize(target), _standardize(regressor)
    beta = float(np.mean(t * r))  # = corr for standardized inputs
    return t - beta * r


def _nonlinear_dependence(a: np.ndarray, b: np.ndarray) -> float:
    """A nonlinear dependence score between ``a`` and ``b`` (0 ⇒ independent).

    Linear covariance of an OLS residual with its regressor is 0 by construction, so
    orientation needs *higher-order* structure: squared/cubed and tanh cross-covariances
    reveal the residual↔cause dependence that betrays the wrong direction.
    """
    a, b = _standardize(a), _standardize(b)
    terms = (
        np.mean(a * b),
        np.mean((a**2 - 1.0) * b),
        np.mean(a * (b**2 - 1.0)),
        np.mean((a**2 - 1.0) * (b**2 - 1.0)),
        np.mean(np.tanh(a) * b) - np.mean(np.tanh(a)) * np.mean(b),
    )
    return float(sum(t * t for t in terms))


def lingam_pairwise_direction(
    x: np.ndarray, y: np.ndarray, *, min_confidence: float = 0.05
) -> tuple[str, float] | None:
    """Return ``("x->y"|"y->x", confidence)`` or ``None`` when unidentifiable.

    ``None`` ⇒ both variables ~Gaussian (LiNGAM not identifiable) or the two directions
    are too symmetric to call.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return None
    x, y = x[mask], y[mask]
    if not (is_non_gaussian(x) or is_non_gaussian(y)):
        return None  # Gaussian → direction not identifiable (report undirected, honestly)

    dep_xy = _nonlinear_dependence(x, _residual(y, x))  # small if X→Y (noise ⊥ cause X)
    dep_yx = _nonlinear_dependence(y, _residual(x, y))  # small if Y→X
    total = dep_xy + dep_yx
    if total < _EPS:
        return None
    confidence = abs(dep_yx - dep_xy) / total
    if confidence < min_confidence:
        return None
    return ("x->y" if dep_xy < dep_yx else "y->x", confidence)


def _partial_correlation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Correlation of ``a`` and ``b`` after linearly removing ``c``."""
    ra, rb = _residual(a, c), _residual(b, c)
    denom = (ra.std() * rb.std()) + _EPS
    return float(np.mean(ra * rb) / denom)


def is_collider(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, *,
    indep_threshold: float = 0.1, dep_threshold: float = 0.2,
) -> bool:
    """A→C←B when the parents A,B are marginally independent but dependent given C."""
    a, b, c = _standardize(a), _standardize(b), _standardize(c)
    marginal = abs(float(np.mean(a * b)))
    conditional = abs(_partial_correlation(a, b, c))
    return marginal < indep_threshold and conditional > dep_threshold


def orient_edge(record: LinkRecord, data: dict[str, np.ndarray]) -> LinkRecord:
    """Orient one edge where licensed; annotate the licensing check.

    Lagged edges are already directed by time. Contemporaneous edges are oriented by
    non-Gaussian identification when possible (source ← cause), else left undirected
    with an explicit ``orientation_undirected_*`` tag.
    """
    if record.edge_type == "lagged_directed":
        # Directed by time precedence — a Rung-1 claim (§IV): past→present is machine-checkable.
        return replace(record, causal_rung=1,
                       causal_assumptions=record.causal_assumptions + ("time_precedence",),
                       warnings=record.warnings + ("directed_by_time_precedence",))
    if record.edge_type != "contemporaneous":
        # latent_shared / nonlinear_residual / mechanical_overlap: associational (Rung-0).
        return replace(record, causal_rung=record.causal_rung if record.causal_rung is not None else 0)
    x = data.get(record.source_var)
    y = data.get(record.target_var)
    if x is None or y is None:
        return replace(record, causal_rung=0)
    result = lingam_pairwise_direction(x, y)
    if result is None:
        # Undirected: the LDO associational baseline (Rung-0), no orientation assumption met.
        return replace(record, causal_rung=0,
                       warnings=record.warnings + ("orientation_undirected_unidentifiable",))
    direction, confidence = result
    if direction == "y->x":  # cause is the current target → swap so source = cause
        record = replace(record, source_var=record.target_var, target_var=record.source_var)
    return replace(
        record, causal_rung=1,
        causal_assumptions=record.causal_assumptions + ("non_gaussian_lingam",),
        warnings=record.warnings + ("oriented_non_gaussian_lingam", f"orientation_confidence_{confidence:.2f}"),
    )


def apply_collider_orientation(
    records: list[LinkRecord], data: dict[str, np.ndarray]
) -> list[LinkRecord]:
    """Rung-1 collider/v-structure orientation (§IV): for every unshielded triple A–C–B (A,B
    both linked to C but NOT to each other), if A⊥B marginally yet A⊥̸B|C, orient A→C←B.

    Operates over the contemporaneous skeleton; each parent edge is directed into the collider
    C and marked Rung-1 with the ``collider_v_structure`` assumption. Machine-checkable, so it
    is auto-applied (unlike Rung-3). Conflicting triples resolve to the last consistent
    orientation — a conservative first pass, not a full PC constraint propagation."""
    from collections import defaultdict

    neighbours: dict[str, set[str]] = defaultdict(set)
    edge_by_pair: dict[frozenset[str], int] = {}
    for idx, r in enumerate(records):
        if r.edge_type == "contemporaneous":
            neighbours[r.source_var].add(r.target_var)
            neighbours[r.target_var].add(r.source_var)
            edge_by_pair[frozenset((r.source_var, r.target_var))] = idx
    oriented: dict[int, LinkRecord] = {}
    for c in sorted(neighbours):
        nbrs = sorted(neighbours[c])
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                if b in neighbours[a]:  # A–B adjacent → shielded triple, not a v-structure
                    continue
                da, db, dc = data.get(a), data.get(b), data.get(c)
                if da is None or db is None or dc is None:
                    continue
                if not is_collider(da, db, dc):
                    continue
                for parent in (a, b):  # orient parent → C (both point into the collider)
                    idx = edge_by_pair.get(frozenset((parent, c)))
                    if idx is None:
                        continue
                    r = oriented.get(idx, records[idx])
                    r = replace(
                        r, source_var=parent, target_var=c, causal_rung=1,
                        causal_assumptions=tuple(dict.fromkeys(r.causal_assumptions + ("collider_v_structure",))),
                        warnings=r.warnings + ("oriented_collider",),
                    )
                    oriented[idx] = r
    return [oriented.get(idx, r) for idx, r in enumerate(records)]


def orient_links(records: list[LinkRecord], data: dict[str, np.ndarray]) -> list[LinkRecord]:
    """Apply Rung-1 orientation to every edge (§IV): pairwise non-Gaussian LiNGAM, then
    collider/v-structure detection over the contemporaneous skeleton."""
    return apply_collider_orientation([orient_edge(r, data) for r in records], data)


__all__ = [
    "is_non_gaussian",
    "lingam_pairwise_direction",
    "is_collider",
    "orient_edge",
    "apply_collider_orientation",
    "orient_links",
]
