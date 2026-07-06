"""CAUSAL-01 — Rung-1 non-Gaussian orientation + collider detection (MSD-III §IV)."""

from __future__ import annotations

import numpy as np

from pegasus.causal.orient import is_collider, orient_edge
from pegasus.ldo.records import LinkRecord


def _non_gaussian_cause_effect(seed: int, n: int = 800):
    rng = np.random.default_rng(seed)
    a = rng.exponential(1.0, n) - 1.0                 # non-Gaussian cause
    b = 0.8 * a + (rng.exponential(1.0, n) - 1.0)     # additive non-Gaussian noise
    return a, b


def test_non_gaussian_orientation_recovers_direction() -> None:
    a, b = _non_gaussian_cause_effect(1)
    data = {"A": a, "B": b}

    out = orient_edge(LinkRecord("A", "B", "contemporaneous", weight=0.7), data)
    assert out.source_var == "A" and out.target_var == "B"      # cause → effect
    assert "oriented_non_gaussian_lingam" in out.warnings

    # a record whose labels are the wrong way round is swapped back to cause→effect
    swapped = orient_edge(LinkRecord("B", "A", "contemporaneous", weight=0.7), data)
    assert swapped.source_var == "A" and swapped.target_var == "B"


def test_gaussian_edge_left_undirected() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(800)
    b = 0.8 * a + rng.standard_normal(800)            # Gaussian → direction unidentifiable
    out = orient_edge(LinkRecord("A", "B", "contemporaneous"), {"A": a, "B": b})
    assert "orientation_undirected_unidentifiable" in out.warnings
    assert "oriented_non_gaussian_lingam" not in out.warnings


def test_collider_detected_but_not_chain() -> None:
    rng = np.random.default_rng(2)
    n = 800
    a = rng.exponential(1, n) - 1
    b = rng.exponential(1, n) - 1
    c = a + b + 0.3 * (rng.exponential(1, n) - 1)     # A→C←B
    assert is_collider(a, b, c)

    # a chain A→C→B is NOT a collider on C (A and B are marginally dependent)
    ca = rng.exponential(1, n) - 1
    cc = ca + 0.3 * rng.standard_normal(n)
    cb = cc + 0.3 * rng.standard_normal(n)
    assert not is_collider(ca, cb, cc)


def test_lagged_edge_directed_by_time() -> None:
    out = orient_edge(LinkRecord("A", "B", "lagged_directed", lag_k=3), {})
    assert "directed_by_time_precedence" in out.warnings
