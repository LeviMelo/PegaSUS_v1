"""HSIC consolidation + CPU/GPU feature-map parity (LDO-B03/B04 + GPU path).

Guards three things: (1) the single public HSIC kernel API lives in hsic.py and the residual
scan routes through it (the duplicate private kernel/repr fns are gone) with a pinned fixed-seed
output; (2) the float32 GPU feature-map path reproduces the CPU statistic (~1e-5 rel) and the
fixed-seed permutation p-value exactly; (3) HSIC detects a nonlinear dependence and clears an
independent pair. The CPU path always runs; GPU assertions are skipped without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

import pegasus.ldo.residual_scan as rs
from pegasus.ldo.hsic import (
    build_hsic_representation,
    hsic_mode_for_n,
    hsic_pair_stat_and_null,
    hsic_pair_stat_and_null_gpu,
)
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.residual_scan import scan_residual_nonlinear_edges


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _nonlinear_field(n_uf=6, per_uf=20, T=4, seed=7):
    rng = np.random.default_rng(seed)
    S = n_uf * per_uf
    uf = np.repeat(np.arange(n_uf), per_uf)
    Z = np.empty((3, S, T))
    A = rng.standard_normal((S, T))
    Z[0] = A
    Z[1] = A**2 + 0.1 * rng.standard_normal((S, T))  # nonlinear A->B, invisible to a linear fit
    Z[2] = rng.standard_normal((S, T))
    space_ids = tuple(f"{27 + uf[s]}{s:05d}"[:7] for s in range(S))
    return GaussianField(variables=("A", "B", "N"), space_ids=space_ids,
                         time_ids=tuple(range(T)), Z=Z, W=np.ones_like(Z), resolution="year")


def test_residual_scan_uses_single_public_hsic_api_no_duplicate_kernel():
    # The reimplemented private kernel/repr fns are gone; the public builder is what runs.
    assert not hasattr(rs, "_centered_kernel")
    src = rs.scan_residual_nonlinear_edges.__module__
    assert build_hsic_representation.__module__ == "pegasus.ldo.hsic"
    assert rs.build_hsic_representation is build_hsic_representation
    assert rs.hsic_pair_stat_and_null is hsic_pair_stat_and_null

    # Fixed-seed scan output is unchanged vs the captured reference (exact mode).
    recs = scan_residual_nonlinear_edges(_nonlinear_field(), np.eye(3), permutations=200, seed=0)
    ab = [r for r in recs if tuple(sorted((r.source_var, r.target_var))) == ("A", "B")]
    assert ab, "expected the planted A-B nonlinear edge"
    assert ab[0].weight == pytest.approx(0.0337958177, abs=1e-9)
    assert ab[0].uncertainty == pytest.approx(0.0054945055, abs=1e-9)
    assert src == "pegasus.ldo.residual_scan"


def test_feature_hsic_detects_nonlinear_and_clears_independent():
    rng = np.random.default_rng(42)
    n = 6000  # > max_exact -> feature-map (rff) mode
    x = rng.standard_normal(n)
    y_dep = x**2 + 0.3 * rng.standard_normal(n)
    y_ind = rng.standard_normal(n)
    mode = hsic_mode_for_n(n=n, budget="fast")
    assert mode == "rff"
    perms = [rng.permutation(n) for _ in range(60)]

    def _pval(y):
        rx = build_hsic_representation(x, mode=mode, kernel="rbf", seed=0, budget="fast", n=n)
        ry = build_hsic_representation(y, mode=mode, kernel="rbf", seed=0, budget="fast", n=n)
        stat, null = hsic_pair_stat_and_null(rx, ry, n=n, perms=perms)
        return (1 + int((null >= stat).sum())) / (1 + null.size)

    assert _pval(y_dep) < 0.05  # nonlinear dependence detected
    assert _pval(y_ind) > 0.20  # independent pair not flagged


def test_cpu_gpu_feature_hsic_parity():
    rng = np.random.default_rng(42)
    n = 6000
    x = rng.standard_normal(n)
    y = x**2 + 0.3 * rng.standard_normal(n)
    mode = hsic_mode_for_n(n=n, budget="fast")
    perms = [rng.permutation(n) for _ in range(60)]
    rx = build_hsic_representation(x, mode=mode, kernel="rbf", seed=0, budget="fast", n=n)
    ry = build_hsic_representation(y, mode=mode, kernel="rbf", seed=0, budget="fast", n=n)

    stat_cpu, null_cpu = hsic_pair_stat_and_null(rx, ry, n=n, perms=perms)
    p_cpu = (1 + int((null_cpu >= stat_cpu).sum())) / (1 + null_cpu.size)
    assert stat_cpu > 0.0

    if not _cuda():
        pytest.skip("CUDA unavailable — CPU path validated; GPU parity assertions skipped")

    stat_gpu, null_gpu = hsic_pair_stat_and_null_gpu(rx["fx"], ry["fy"], n=n, perms=perms, seed=0)
    p_gpu = (1 + int((null_gpu >= stat_gpu).sum())) / (1 + null_gpu.size)
    assert stat_gpu == pytest.approx(stat_cpu, rel=1e-5)
    assert p_gpu == p_cpu  # identical exceedance count over the shared permutation set
