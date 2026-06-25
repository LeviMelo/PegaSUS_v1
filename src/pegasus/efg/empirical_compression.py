"""Stage-2 post-selection empirical equivalence compression (MSD §3.15.2).

Stage-1 topological precompression (``efg/equivalence.py``) suppresses fields
that are *provably* identical from metadata alone. Stage-2 catches fields that
are empirically indistinguishable on the materialized data:

    C_emp(X_i, X_j) = max(|rho_Pearson|, |rho_Spearman|)

When ``C_emp >= 0.98`` the lower-utility node is folded into the higher-utility
dominant node of its equivalence class. Per the MSD, **no data are destroyed** —
the suppressed nodes remain in V_fields / Q_tensor / E_DAG; this report only
records the canonical mapping so PIRS does not spend its Top-K budget on
empirically duplicate covariates.

This runs only after candidate pruning by utility/budget/TopK on materialized
value vectors (MSD §7: never on metadata-only or planned nodes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DEFAULT_THRESHOLD = 0.98
MIN_PAIRWISE_OBS = 3


@dataclass(frozen=True)
class SuppressedEmpirical:
    suppressed_field_id: str
    canonical_field_id: str
    c_emp: float
    pearson: float
    spearman: float
    n_pairwise: int

    def as_manifest(self) -> dict[str, object]:
        return {
            "suppressed_field_id": self.suppressed_field_id,
            "canonical_field_id": self.canonical_field_id,
            "c_emp": self.c_emp,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "n_pairwise": self.n_pairwise,
            "proof": "empirical_correlation_ge_threshold",
        }


@dataclass(frozen=True)
class EmpiricalCompressionReport:
    threshold: float
    input_count: int
    output_count: int
    suppressed: tuple[SuppressedEmpirical, ...]
    canonical_by_field_id: dict[str, str]
    skipped_field_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def suppressed_count(self) -> int:
        return len(self.suppressed)

    def as_manifest(self) -> dict[str, object]:
        return {
            "stage": "empirical_post_selection_compression",
            "threshold": self.threshold,
            "uses_numerical_arrays": True,
            "data_destroyed": False,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "suppressed_count": self.suppressed_count,
            "suppressed": [item.as_manifest() for item in self.suppressed],
            "canonical_by_field_id": dict(self.canonical_by_field_id),
            "skipped_field_ids": list(self.skipped_field_ids),
        }


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so Spearman = Pearson of ranks."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # average tied ranks
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = np.mean(ranks[order[i:j]])
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        # A constant column cannot be empirically equivalent to anything by
        # correlation; Stage-1 zero-variance handling owns that case.
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def empirical_c_emp(x: Sequence[float | None], y: Sequence[float | None]) -> tuple[float, float, float, int]:
    """Return (c_emp, pearson, spearman, n_pairwise) over complete pairs."""
    xa = np.asarray([np.nan if v is None else float(v) for v in x], dtype=float)
    ya = np.asarray([np.nan if v is None else float(v) for v in y], dtype=float)
    n = min(len(xa), len(ya))
    xa, ya = xa[:n], ya[:n]
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[mask], ya[mask]
    if xa.size < MIN_PAIRWISE_OBS:
        return 0.0, 0.0, 0.0, int(xa.size)
    pearson = _safe_corr(xa, ya)
    spearman = _safe_corr(_rank(xa), _rank(ya))
    c_emp = max(abs(pearson), abs(spearman))
    return c_emp, pearson, spearman, int(xa.size)


def empirical_compress(
    items: Sequence[tuple[str, float, Sequence[float | None] | None]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> EmpiricalCompressionReport:
    """Fold empirically-equivalent fields into higher-utility canonicals.

    ``items`` is ``(field_id, utility, value_vector)``. Fields whose value
    vector is missing are kept as canonical and reported as skipped (they
    cannot be empirically compared).
    """
    # Higher utility first so the dominant node of each class is canonical.
    ordered = sorted(items, key=lambda it: (-float(it[1]), str(it[0])))
    canonical: list[tuple[str, np.ndarray | None]] = []
    canonical_by_id: dict[str, str] = {}
    suppressed: list[SuppressedEmpirical] = []
    skipped: list[str] = []
    input_ids = [str(it[0]) for it in ordered]

    for field_id, _utility, vector in ordered:
        field_id = str(field_id)
        vec = (
            None
            if vector is None
            else np.asarray([np.nan if v is None else float(v) for v in vector], dtype=float)
        )
        if vec is None:
            skipped.append(field_id)
            canonical.append((field_id, None))
            canonical_by_id[field_id] = field_id
            continue
        matched: SuppressedEmpirical | None = None
        for canon_id, canon_vec in canonical:
            if canon_vec is None:
                continue
            c_emp, pearson, spearman, n_pairwise = empirical_c_emp(canon_vec, vec)
            if c_emp >= threshold:
                matched = SuppressedEmpirical(
                    suppressed_field_id=field_id,
                    canonical_field_id=canon_id,
                    c_emp=c_emp,
                    pearson=pearson,
                    spearman=spearman,
                    n_pairwise=n_pairwise,
                )
                break
        if matched is not None:
            suppressed.append(matched)
            canonical_by_id[field_id] = matched.canonical_field_id
        else:
            canonical.append((field_id, vec))
            canonical_by_id[field_id] = field_id

    return EmpiricalCompressionReport(
        threshold=threshold,
        input_count=len(input_ids),
        output_count=len(canonical),
        suppressed=tuple(suppressed),
        canonical_by_field_id=canonical_by_id,
        skipped_field_ids=tuple(skipped),
    )
