"""LDO residual nonlinear-edge layer (MSD-II §II.6.3, MII-LDO-05).

After the structured linear backbone is fit, compute the joint-model residual
field ``e = Z - Ẑ`` (the Gaussian-graphical conditional residual
``e_j = (Ω Z)_j / Ω_jj``) and run the **existing** HSIC scanner (§6.7) on residual
pairs under the §6.8 nulls with §6.9 FDR. HSIC here is a *targeted nonlinear-edge
detector on what the structured backbone cannot explain* — a strict
generalization of MSD §6 that reuses ``pirs/{hsic,nulls,fdr}.py`` unchanged.

A significant residual HSIC between variables the linear precision left
unconnected is a ``nonlinear_residual`` link.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import random as _random

from pegasus.ldo.envelope import (
    ScaleExceedsEnvelopeError,
    estimate_residual_scan_bytes,
    load_compute_envelope,
)
from pegasus.ldo.fdr import correct_p_values
from pegasus.ldo.hsic import (
    _bandwidth,
    _np_nystrom_features,
    _np_rff_features,
)
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.nulls import generate_null_indices
from pegasus.ldo.records import LinkRecord

import math

_MIN_N_EFF = 100
_MIN_NULL_BLOCKS = 5  # < this many spatial blocks → certify descriptive only (§6.8/§6.9)


def _residual_scan_memory_budget() -> int:
    """Bytes the residual-scan representation cache may use before it must refuse.

    Prefer 60% of the *currently available* system RAM (the scan is a CPU/numpy
    allocation, not a VRAM one), leaving headroom for its transient per-variable
    ``n×n`` kernel builds and the rest of the process. Falls back to the compute
    envelope (config/compute.yaml) if psutil is unavailable.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available * 0.60)
    except Exception:
        return int(load_compute_envelope().max_bytes)


@dataclass
class ResidualEdge:
    source: str
    target: str
    hsic: float
    p_value: float
    q_value: float


def joint_model_residuals(Z_matrix: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Gaussian-graphical conditional residuals ``e_j = (Ω Z)_j / Ω_jj`` (p × n)."""
    diag = np.clip(np.diag(precision), 1e-12, None)
    return (precision @ Z_matrix) / diag[:, None]


# --- Per-variable HSIC representation cache (residual-scan fast path, §V.3) -----------
# The exhaustive p²/2 residual HSIC scan re-derived each variable's kernel/feature
# representation once per pair — O(p²) rebuilds of an object that only depends on ONE
# variable. Cache each variable's representation once (O(p) builds); a pair is then a
# cheap product-sum (exact) or feature cross-covariance (approx). Reproduces
# ``numpy_kernel_hsic_permutation_test`` bit-for-bit: same bandwidth seeds (x-side=seed,
# y-side=seed+1), same kernels, same centering, same feature maps.


def _centered_kernel(v: np.ndarray, *, bandwidth: float, kernel: str) -> np.ndarray:
    """Doubly-centered kernel matrix H K H (matches hsic.py exact path)."""
    dist = np.abs(v[:, None] - v[None, :])
    if kernel == "linear":
        K = v[:, None] * v[None, :]
    elif kernel == "matern":
        scaled = math.sqrt(3.0) * dist / bandwidth
        K = (1.0 + scaled) * np.exp(-scaled)
    else:
        K = np.exp(-(dist ** 2) / (2.0 * bandwidth ** 2))
    # H K H == K - rowmean - colmean + grandmean (identical to the eye/full form,
    # avoids materializing the m×m centering matrix H and two m×m matmuls).
    rm = K.mean(axis=0, keepdims=True)
    cm = K.mean(axis=1, keepdims=True)
    return K - rm - cm + K.mean()


def _build_var_reprs(
    E: np.ndarray, *, seed: int, kernel: str, budget: str, max_exact: int
) -> tuple[str, list[dict]]:
    """Precompute each variable's HSIC representation once (x-side and y-side).

    Returns ``(hsic_mode, reprs)`` where ``reprs[v]`` carries the doubly-centered kernel
    (exact mode) or the RFF/Nyström feature maps (approx mode) for variable ``v`` as
    covariate (``seed``) and as residual (``seed+1``) — the two seed regimes the pairwise
    call uses for the x and y slots respectively.
    """
    p, n = E.shape
    reprs: list[dict] = []
    if n <= max_exact:
        for v in range(p):
            row = E[v]
            bw_x = _bandwidth(row.tolist(), seed=seed)
            bw_y = _bandwidth(row.tolist(), seed=seed + 1)
            reprs.append({
                "kx": _centered_kernel(row, bandwidth=bw_x, kernel=kernel),
                "ky": _centered_kernel(row, bandwidth=bw_y, kernel=kernel),
            })
        return "exact", reprs

    n_features = int(min(max(128, int(math.sqrt(n) * 4)), 1024))
    if budget == "fast":
        mode = "rff"
        for v in range(p):
            row = E[v]
            bw_x = _bandwidth(row.tolist(), seed=seed)
            bw_y = _bandwidth(row.tolist(), seed=seed + 1)
            fx = _np_rff_features(row, bandwidth=bw_x, features=n_features, seed=seed)
            fy = _np_rff_features(row, bandwidth=bw_y, features=n_features, seed=seed + 1)
            reprs.append({"fx": fx - fx.mean(axis=0, keepdims=True),
                          "fy": fy - fy.mean(axis=0, keepdims=True)})
    else:
        mode = "nystrom"
        landmarks = int(min(max(64, int(math.sqrt(n))), 1024))
        for v in range(p):
            row = E[v]
            bw_x = _bandwidth(row.tolist(), seed=seed)
            bw_y = _bandwidth(row.tolist(), seed=seed + 1)
            fx = _np_nystrom_features(row, bandwidth=bw_x, landmarks=landmarks, seed=seed)
            fy = _np_nystrom_features(row, bandwidth=bw_y, landmarks=landmarks, seed=seed + 1)
            reprs.append({"fx": fx - fx.mean(axis=0, keepdims=True),
                          "fy": fy - fy.mean(axis=0, keepdims=True)})
    return mode, reprs


def _pair_stat_and_null(
    ri: dict, rj: dict, *, mode: str, n: int, perms: list[np.ndarray] | None,
) -> tuple[float, np.ndarray]:
    """HSIC statistic + null vector for pair (i as covariate, j as residual)."""
    if mode == "exact":
        kx = ri["kx"]
        ky = rj["ky"]
        denom = max((n - 1) ** 2, 1)
        stat = float(np.sum(kx * ky) / denom)
        if perms is None:
            return stat, np.empty(0)
        null = np.array([float(np.sum(kx * ky[np.ix_(pm, pm)]) / denom) for pm in perms])
        return stat, null
    fx = ri["fx"]
    fy = rj["fy"]
    denom = max(n - 1, 1)
    cross = fx.T @ fy / denom
    stat = float((cross * cross).sum())
    if perms is None:
        return stat, np.empty(0)
    null = np.empty(len(perms), dtype=np.float64)
    for k, pm in enumerate(perms):
        c = fx.T @ fy[pm] / denom
        null[k] = float((c * c).sum())
    return stat, null


def scan_residual_nonlinear_edges(
    field: GaussianField,
    precision: np.ndarray,
    *,
    budget: str = "standard",
    permutations: int = 200,
    seed: int = 0,
    alpha: float = 0.1,
) -> list[LinkRecord]:
    """Detect nonlinear residual dependence the linear backbone missed."""
    p, S, T = field.shape
    Z = field.Z.reshape(p, S * T)
    # Complete-case cells (all variables observed) for a shared residual sample.
    observed = np.isfinite(Z).all(axis=0)
    n_eff = int(observed.sum())
    if n_eff < _MIN_N_EFF:
        return []  # §6.7 gate: underpowered → no verified nonlinear scan
    Zc = Z[:, observed]
    Zc = np.where(np.isfinite(Zc), Zc, 0.0)
    E = joint_model_residuals(Zc, precision)

    # Structured null (§6.8): the residuals retain spatial autocorrelation, so an iid
    # permutation null is anticonservative — two variables that merely co-cluster in
    # space (both high in the Northeast) test as dependent. Permute within spatial
    # blocks (UF = first 2 digits of the municipality id of each cell's space unit);
    # this breaks cross-variable dependence while preserving spatial-block identity,
    # testing dependence *beyond* shared clustering. Reuse the same structural
    # permutations across pairs (the null is structural, not data-dependent).
    observed_idx = np.where(observed)[0]
    space_of_cell = (observed_idx // T).tolist()
    uf_labels = [
        str(field.space_ids[s])[:2] if s < len(field.space_ids) else "00"
        for s in space_of_cell
    ]
    n_spatial_blocks = len(set(uf_labels))
    if n_spatial_blocks >= 2:
        rng = _random.Random(seed)
        perm_list: list[list[int]] | None = [
            generate_null_indices(
                strategy="restricted_intra_uf_spatial_swap",
                n=n_eff, support={"uf_strata": uf_labels}, rng=rng,
            )
            for _ in range(permutations)
        ]
        null_strategy = "restricted_intra_uf_spatial_swap"
    else:
        perm_list = None  # no spatial structure (single block) → iid fallback
        null_strategy = "iid_permutation"
    # §6.8/§6.9: certify a discovery only with enough spatial blocks for a valid null;
    # otherwise the edge is reported descriptive (surfaced, not certified).
    sufficient_blocks = n_spatial_blocks >= _MIN_NULL_BLOCKS

    # §II.10 refuse-don't-degrade: the per-variable representation cache below retains
    # p × 2 dense kernels/feature-maps and is the single largest LDO allocation (absent
    # from the fit-path envelope, envelope.estimate_ldo_bytes). At national n_eff it can be
    # tens-to-hundreds of GB; attempting it thrashes swap then raises MemoryError. Pre-check
    # against the actually-available RAM and refuse with a clear reason (the orchestrator
    # catches this into a residual-scan diagnostic — the nonlinear layer is honestly reported
    # as not-computed-at-this-scale rather than silently OOM-dropped after a long thrash).
    _scan_bytes = estimate_residual_scan_bytes(p=p, n_eff=n_eff, budget=budget)
    if _scan_bytes > _residual_scan_memory_budget():
        raise ScaleExceedsEnvelopeError(
            "residual_hsic_scan_exceeds_memory: estimated "
            f"{_scan_bytes / 1024**3:.1f} GB HSIC representation cache "
            f"(p={p}, n_eff={n_eff}, budget={budget}) exceeds the available-memory budget; "
            "nonlinear residual edges not computed at this scale (§II.7 multi-resolution / "
            "§II.10 refuse-don't-degrade)."
        )

    # Precompute each variable's HSIC representation ONCE (O(p) builds), then score every
    # pair from the cache. The old loop re-derived per-variable kernels/features inside
    # each of the p²/2 pairwise calls — an O(p²) rebuild of a per-variable object — and
    # re-ran the 2048-sample bandwidth per side per pair. This is bit-identical to
    # ``numpy_kernel_hsic_permutation_test`` (verified against it in tests): same seed
    # regimes, kernels, centering, feature maps, and the same shared permutation set.
    mode, reprs = _build_var_reprs(E, seed=seed, kernel="rbf", budget=budget, max_exact=5000)

    # Resolve the shared permutation set once (structural, or the iid fallback the pairwise
    # call would have drawn from `seed` — identical across pairs, so drawn a single time).
    if perm_list is not None:
        perms: list[np.ndarray] | None = [
            np.asarray(pm, dtype=int) for pm in perm_list if len(pm) == n_eff
        ]
    else:
        _rng = np.random.default_rng(seed)
        perms = [_rng.permutation(n_eff) for _ in range(max(int(permutations), 1))]

    pairs: list[tuple[int, int]] = [(i, j) for i in range(p) for j in range(i + 1, p)]
    stats: list[float] = []
    pvals: list[float] = []
    for i, j in pairs:
        stat, null_arr = _pair_stat_and_null(reprs[i], reprs[j], mode=mode, n=n_eff, perms=perms)
        pval = float((1 + int((null_arr >= stat).sum())) / (1 + null_arr.size)) if null_arr.size else 1.0
        stats.append(stat)
        pvals.append(pval)

    qvals = correct_p_values(pvals, method="BH").q_values if pvals else []
    records: list[LinkRecord] = []
    for (i, j), stat, pv, qv in zip(pairs, stats, pvals, qvals):
        if qv is not None and qv <= alpha:
            records.append(
                LinkRecord(
                    source_var=field.variables[i],
                    target_var=field.variables[j],
                    edge_type="nonlinear_residual",
                    weight=float(stat),
                    uncertainty=float(qv),
                    null_strategy=null_strategy,
                    fdr_method="benjamini_hochberg",
                    certification_status="selected" if sufficient_blocks else "descriptive",
                )
            )
    return records


__all__ = ["ResidualEdge", "joint_model_residuals", "scan_residual_nonlinear_edges"]
