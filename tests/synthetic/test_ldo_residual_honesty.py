"""O10 / O11 — residual-scan provenance truthfulness + loud refusal (no silent no-op).

O10: the stamped ``null_strategy`` must name the permutation that ACTUALLY ran, never a
season-preserving/cyclic-shift generator that did not execute (a provenance falsification).
O11: a compute-envelope refusal must PROPAGATE (not be swallowed into a diagnostic string), and
an underpowered complete-case sample must RAISE a typed skip (not silently return no edges).
"""

from __future__ import annotations

import numpy as np
import pytest

import pegasus.ldo.residual_scan as rs
from pegasus.ldo.envelope import ScaleExceedsEnvelopeError
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.residual_scan import (
    ResidualScanUnderpowered,
    scan_residual_nonlinear_edges,
)


def _field(n_uf=6, per_uf=20, T=4, seed=11):
    rng = np.random.default_rng(seed)
    S = n_uf * per_uf
    uf = np.repeat(np.arange(n_uf), per_uf)
    eff = rng.standard_normal(n_uf) * 2.0
    Z = np.empty((3, S, T))
    for s in range(S):
        Z[0, s, :] = eff[uf[s]] + rng.standard_normal(T) * 0.4
        Z[1, s, :] = eff[uf[s]] * 1.8 + rng.standard_normal(T) * 0.4
        Z[2, s, :] = rng.standard_normal(T)
    space_ids = tuple(f"{27 + uf[s]}{s:05d}"[:7] for s in range(S))
    return GaussianField(variables=("A", "B", "N"), space_ids=space_ids,
                         time_ids=tuple(range(T)), Z=Z, W=np.ones_like(Z), resolution="year")


def _field_with_real_nonlinear_edge(n_uf=6, per_uf=20, T=4, seed=7):
    """A and B carry a genuine within-cell nonlinear dependence (B ≈ A²) that survives the
    structured null, so ≥1 nonlinear_residual edge is emitted to inspect its provenance."""
    rng = np.random.default_rng(seed)
    S = n_uf * per_uf
    uf = np.repeat(np.arange(n_uf), per_uf)
    Z = np.empty((3, S, T))
    A = rng.standard_normal((S, T))
    Z[0] = A
    Z[1] = A ** 2 + 0.1 * rng.standard_normal((S, T))     # nonlinear A→B (invisible to a linear fit)
    Z[2] = rng.standard_normal((S, T))
    space_ids = tuple(f"{27 + uf[s]}{s:05d}"[:7] for s in range(S))
    return GaussianField(variables=("A", "B", "N"), space_ids=space_ids,
                         time_ids=tuple(range(T)), Z=Z, W=np.ones_like(Z), resolution="year")


def test_null_strategy_names_the_executed_permutation_not_a_dead_generator():
    records = scan_residual_nonlinear_edges(
        _field_with_real_nonlinear_edge(), np.eye(3), permutations=200, seed=0)
    assert records
    for r in records:
        # the executed null is the within-block restricted swap — the label must say so
        assert r.null_strategy.startswith("restricted_within_spatial_block")
        # and MUST NOT claim a circular-shift generator that never ran (the O10 falsification)
        assert "season_preserving" not in r.null_strategy
        assert "moving_block_circular_shift" not in r.null_strategy
        # the panel-type regime that parameterized the strata is recorded honestly
        assert "regime:annual_municipal_panel" in r.null_strategy


def test_underpowered_complete_case_raises_typed_skip_not_silent_empty():
    # only 60 complete cells (< the §6.7 power floor) → a typed skip, not a silent []
    small = _field(n_uf=3, per_uf=5, T=4)   # S=15, T=4 → 60 cells
    with pytest.raises(ResidualScanUnderpowered):
        scan_residual_nonlinear_edges(small, np.eye(3), permutations=50, seed=0)


def test_envelope_refusal_propagates_not_swallowed(monkeypatch):
    # force even the coarsest grain over budget → the scan must RAISE ScaleExceedsEnvelopeError
    monkeypatch.setattr(rs, "_residual_scan_memory_budget", lambda: 1)  # 1 byte
    with pytest.raises(ScaleExceedsEnvelopeError):
        scan_residual_nonlinear_edges(_field(), np.eye(3), permutations=50, seed=0)


def test_run_ldo_reraises_envelope_refusal(monkeypatch):
    from pegasus.ldo.orchestrator import run_ldo
    monkeypatch.setattr(rs, "_residual_scan_memory_budget", lambda: 1)
    with pytest.raises(ScaleExceedsEnvelopeError):
        run_ldo(_field(), K=1, run_residual_scan=True, enforce_envelope=False,
                n_subsamples=2, seed=0)
