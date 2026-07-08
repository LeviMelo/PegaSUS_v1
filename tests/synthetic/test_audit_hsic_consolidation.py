"""ADVERSARIAL AUDIT of commit 53af933 (HSIC kernel consolidation + float32 GPU path).

Refutes the commit's claims:
  CLAIM-A "CPU path byte-identical (pinned residual-scan edge unchanged)."
  CLAIM-B "GPU float32 path matches CPU ... permutation p-values identical."

The commit's own byte-identity evidence is a SINGLE pinned exact-mode edge
(tests/synthetic/test_hsic_gpu_parity.py). These tests attack the parts that pin
does NOT cover:

  1. Full live-scan output (every edge's stat + p-value), not one pinned edge,
     reconstructing the pre-consolidation ``scan_residual_nonlinear_edges`` and
     comparing it to the current one — in BOTH exact and feature-map modes.

  2. The feature-map LIVE path on a CUDA machine: the current
     ``scan_residual_nonlinear_edges`` silently routes feature-map pairs to the
     float32 GPU kernel, so the live weight is NOT float64-identical to the
     pre-consolidation numpy path. This DIRECTLY refutes a literal reading of
     "CPU path byte-identical" for the live scan whenever CUDA is present.

  3. The GPU p-value robustness claim, probed adversarially on independent pairs
     (statistic sits inside the null cloud — maximally sensitive to float32
     rounding of the exceedance count).

The pre-consolidation implementation is reconstructed VERBATIM from commit
53af933^ (it depended only on ``_bandwidth`` / ``_np_rff_features`` /
``_np_nystrom_features``, which still exist in the current hsic.py), so the
comparison isolates exactly what the consolidation changed.
"""

from __future__ import annotations

import math
import random as _random

import numpy as np
import pytest

from pegasus.ldo.hsic import _bandwidth, _np_nystrom_features, _np_rff_features
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.nulls import generate_null_indices, select_null_regime
from pegasus.ldo.residual_scan import (
    joint_model_residuals,
    scan_residual_nonlinear_edges,
)
from pegasus.ldo.records import LinkRecord
from pegasus.ldo.fdr import correct_p_values


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Pre-consolidation (53af933^) private HSIC repr + pair kernel, verbatim.
# --------------------------------------------------------------------------- #
def _old_centered_kernel(v: np.ndarray, *, bandwidth: float, kernel: str) -> np.ndarray:
    dist = np.abs(v[:, None] - v[None, :])
    if kernel == "linear":
        K = v[:, None] * v[None, :]
    elif kernel == "matern":
        scaled = math.sqrt(3.0) * dist / bandwidth
        K = (1.0 + scaled) * np.exp(-scaled)
    else:
        K = np.exp(-(dist ** 2) / (2.0 * bandwidth ** 2))
    rm = K.mean(axis=0, keepdims=True)
    cm = K.mean(axis=1, keepdims=True)
    return K - rm - cm + K.mean()


def _old_build_var_reprs(E, *, seed, kernel, budget, max_exact):
    p, n = E.shape
    reprs = []
    if n <= max_exact:
        for v in range(p):
            row = E[v]
            bw_x = _bandwidth(row.tolist(), seed=seed)
            bw_y = _bandwidth(row.tolist(), seed=seed + 1)
            reprs.append({"kx": _old_centered_kernel(row, bandwidth=bw_x, kernel=kernel),
                          "ky": _old_centered_kernel(row, bandwidth=bw_y, kernel=kernel)})
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


def _old_pair_stat_and_null(ri, rj, *, mode, n, perms):
    if mode == "exact":
        kx, ky = ri["kx"], rj["ky"]
        denom = max((n - 1) ** 2, 1)
        stat = float(np.sum(kx * ky) / denom)
        if perms is None:
            return stat, np.empty(0)
        null = np.array([float(np.sum(kx * ky[np.ix_(pm, pm)]) / denom) for pm in perms])
        return stat, null
    fx, fy = ri["fx"], rj["fy"]
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


def _old_scan(field: GaussianField, precision, *, budget, permutations, seed, alpha=0.1):
    """Pre-consolidation live scan, with the current scan's structured-null/coarsen
    plumbing but the OLD (numpy-only, float64) repr + pair kernel. Isolates exactly
    the kernel-routing change the consolidation introduced."""
    p_all, S, T = field.shape
    Z_all = field.Z.reshape(p_all, S * T)
    observed_full = np.isfinite(Z_all).all(axis=0)
    kept = np.arange(p_all)
    if int(observed_full.sum()) < 100 and p_all > 2:
        order = np.argsort(-np.isfinite(Z_all).mean(axis=1))
        for cut in range(p_all, 1, -1):
            sub = np.sort(order[:cut])
            if int(np.isfinite(Z_all[sub]).all(axis=0).sum()) >= 100:
                kept = sub
                break
    variables = [field.variables[k] for k in kept]
    p = len(kept)
    precision = np.asarray(precision)[np.ix_(kept, kept)]
    Z = Z_all[kept]
    observed = np.isfinite(Z).all(axis=0)
    Zc = np.where(np.isfinite(Z[:, observed]), Z[:, observed], 0.0)
    E = joint_model_residuals(Zc, precision)

    from pegasus.ldo.residual_scan import _infer_panel_type, _temporal_bucket

    panel_type = _infer_panel_type(field)
    regime = select_null_regime(panel_type)
    permutations = int(regime.permutations)
    observed_idx = np.where(observed)[0]
    uf_of_cell = np.array([
        str(field.space_ids[s])[:2] if s < len(field.space_ids) else "00"
        for s in (observed_idx // T)
    ])
    bucket_of_cell = _temporal_bucket(observed_idx % T, panel_type)
    n = int(E.shape[1])
    n_spatial_blocks = len(set(uf_of_cell.tolist()))
    n_temporal_blocks = len(set(bucket_of_cell.tolist()))
    strata = [f"{u}|{b}" for u, b in zip(uf_of_cell.tolist(), bucket_of_cell.tolist())]
    if n_spatial_blocks >= 2 or n_temporal_blocks >= 2:
        rng = _random.Random(seed)
        perm_list = [
            generate_null_indices(strategy="restricted_intra_uf_spatial_swap",
                                  n=n, support={"uf_strata": strata}, rng=rng)
            for _ in range(permutations)
        ]
    else:
        perm_list = None

    mode, reprs = _old_build_var_reprs(E, seed=seed, kernel="rbf", budget=budget, max_exact=5000)
    if perm_list is not None:
        perms = [np.asarray(pm, dtype=int) for pm in perm_list if len(pm) == n]
    else:
        _rng = np.random.default_rng(seed)
        perms = [_rng.permutation(n) for _ in range(max(permutations, 1))]

    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    stats, pvals = [], []
    for i, j in pairs:
        stat, null_arr = _old_pair_stat_and_null(reprs[i], reprs[j], mode=mode, n=n, perms=perms)
        pv = float((1 + int((null_arr >= stat).sum())) / (1 + null_arr.size)) if null_arr.size else 1.0
        stats.append(stat)
        pvals.append(pv)
    qvals = correct_p_values(pvals, method=regime.fdr_method).q_values if pvals else []
    out = {}
    for (i, j), stat, pv, qv in zip(pairs, stats, pvals, qvals):
        if qv is not None and qv <= alpha:
            out[tuple(sorted((variables[i], variables[j])))] = (stat, qv)
    return out


# --------------------------------------------------------------------------- #
# Fields.
# --------------------------------------------------------------------------- #
def _exact_field(n_uf=6, per_uf=25, T=4, seed=7, p=4):
    rng = np.random.default_rng(seed)
    S = n_uf * per_uf
    uf = np.repeat(np.arange(n_uf), per_uf)
    Z = np.empty((p, S, T))
    A = rng.standard_normal((S, T))
    Z[0] = A
    Z[1] = A ** 2 + 0.1 * rng.standard_normal((S, T))
    for k in range(2, p):
        Z[k] = rng.standard_normal((S, T))
    space_ids = tuple(f"{27 + uf[s]}{s:05d}"[:7] for s in range(S))
    return GaussianField(variables=tuple(chr(65 + i) for i in range(p)), space_ids=space_ids,
                         time_ids=tuple(range(T)), Z=Z, W=np.ones_like(Z), resolution="year")


def _featuremap_field(n_uf=6, per_uf=1100, T=1, seed=2, p=3):
    rng = np.random.default_rng(seed)
    S = n_uf * per_uf
    uf = np.repeat(np.arange(n_uf), per_uf)
    Z = np.empty((p, S, T))
    A = rng.standard_normal((S, T))
    Z[0] = A
    Z[1] = A ** 2 + 0.1 * rng.standard_normal((S, T))
    for k in range(2, p):
        Z[k] = rng.standard_normal((S, T))
    space_ids = tuple(f"{27 + uf[s]}{s:06d}"[:7] for s in range(S))
    return GaussianField(variables=tuple(chr(65 + i) for i in range(p)), space_ids=space_ids,
                         time_ids=tuple(range(T)), Z=Z, W=np.ones_like(Z), resolution="year")


# --------------------------------------------------------------------------- #
# Test 1 — exact-mode live scan byte-identity across seeds (should PASS: the
# consolidation preserved the exact CPU path; the pinned edge generalizes).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 3])
def test_exact_mode_live_scan_byte_identical(seed):
    field = _exact_field(seed=seed)
    prec = np.eye(len(field.variables))
    new = scan_residual_nonlinear_edges(field, prec, permutations=200, seed=seed)
    new_map = {tuple(sorted((r.source_var, r.target_var))): (r.weight, r.uncertainty) for r in new}
    old_map = _old_scan(field, prec, budget="standard", permutations=200, seed=seed)
    assert set(new_map) == set(old_map), (set(new_map), set(old_map))
    for key in new_map:
        assert new_map[key][0] == old_map[key][0], f"stat drift on {key}: {new_map[key][0]} vs {old_map[key][0]}"
        assert new_map[key][1] == old_map[key][1], f"pval drift on {key}"


# --------------------------------------------------------------------------- #
# Test 2 — feature-map LIVE scan: refute "CPU path byte-identical" on CUDA.
# The current scan routes feature-map pairs to the float32 GPU kernel, so the
# live weight is NOT float64-identical to the pre-consolidation numpy weight.
# On a CUDA host this test FAILS the strict byte-identity, then documents that
# the divergence is float32-scale (~1e-6 rel) rather than a logic error.
# --------------------------------------------------------------------------- #
def test_featuremap_live_scan_byte_identity_or_float32_only():
    field = _featuremap_field()
    prec = np.eye(len(field.variables))
    n_eff = int(np.isfinite(field.Z.reshape(field.shape[0], -1)).all(axis=0).sum())
    assert n_eff > 5000, "field must exercise feature-map mode"

    new = scan_residual_nonlinear_edges(field, prec, permutations=100, seed=0)
    new_map = {tuple(sorted((r.source_var, r.target_var))): (r.weight, r.uncertainty) for r in new}
    old_map = _old_scan(field, prec, budget="standard", permutations=100, seed=0)
    assert set(new_map) == set(old_map) and new_map, (set(new_map), set(old_map))

    strict_identical = all(new_map[k][0] == old_map[k][0] for k in new_map)
    if _cuda():
        # ADVERSARIAL ASSERTION: the commit says "CPU path byte-identical". On a CUDA
        # machine the LIVE feature-map scan is NOT float64-identical — it silently used
        # the float32 GPU kernel. This is the refutation of the literal claim.
        assert not strict_identical, (
            "Feature-map live scan was byte-identical on CUDA — GPU float32 path may "
            "not be engaging (claim would then be trivially true but GPU path dead)."
        )
        # But the divergence must be float32-scale only (else it is a real logic bug):
        for k in new_map:
            rel = abs(new_map[k][0] - old_map[k][0]) / max(abs(old_map[k][0]), 1e-30)
            assert rel < 1e-5, f"feature-map divergence on {k} exceeds float32 tolerance: rel={rel:.2e}"
    else:
        # No CUDA -> the numpy feature path runs; it MUST be exactly float64-identical.
        assert strict_identical, "feature-map CPU path diverged without CUDA"


# --------------------------------------------------------------------------- #
# Test 3 — GPU p-value robustness on the maximally sensitive case (independent
# pairs: statistic sits inside the null cloud). Refutes/confirms "p-values
# identical". Skipped without CUDA.
# --------------------------------------------------------------------------- #
def test_gpu_pvalue_matches_cpu_on_independent_pairs():
    if not _cuda():
        pytest.skip("CUDA unavailable")
    from pegasus.ldo.hsic import (
        build_hsic_representation,
        hsic_mode_for_n,
        hsic_pair_stat_and_null,
        hsic_pair_stat_and_null_gpu,
    )

    n = 6000
    flips = 0
    for trial in range(4):
        rng = np.random.default_rng(3000 + trial)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)  # independent: p-value maximally rounding-sensitive
        mode = hsic_mode_for_n(n=n, budget="fast")
        rx = build_hsic_representation(x, mode=mode, kernel="rbf", seed=0, budget="fast", n=n)
        ry = build_hsic_representation(y, mode=mode, kernel="rbf", seed=0, budget="fast", n=n)
        perms = [rng.permutation(n) for _ in range(60)]
        s_c, nu_c = hsic_pair_stat_and_null(rx, ry, n=n, perms=perms)
        s_g, nu_g = hsic_pair_stat_and_null_gpu(rx["fx"], ry["fy"], n=n, perms=perms, seed=0)
        p_c = (1 + int((nu_c >= s_c).sum())) / (1 + nu_c.size)
        p_g = (1 + int((nu_g >= s_g).sum())) / (1 + nu_g.size)
        rel = abs(s_g - s_c) / max(abs(s_c), 1e-30)
        assert rel < 1e-5, f"trial {trial}: statistic parity worse than float32: rel={rel:.2e}"
        if p_c != p_g:
            flips += 1
    assert flips == 0, f"{flips}/4 GPU p-values disagreed with CPU (float32 exceedance flip)"
