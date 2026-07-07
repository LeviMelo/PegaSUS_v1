"""O5 / §IX.3 — temporal holdout (fit-through-T, verify-T+1) + Place×Time stability perturbation.

A real edge persists on the held-out tail; a fluke evaporates. And stability selection must
perturb the TIME axis too, not space alone — an edge that survives only one time span is filtered.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.edges import stability_select
from pegasus.ldo.margins import GaussianField
from pegasus.validation.holdout import temporal_holdout


def _persistent_edge_field(seed=0, T=12):
    rng = np.random.default_rng(seed)
    p, S = 4, 60
    Z = rng.standard_normal((p, S, T))
    Z[1] += 0.8 * Z[0]          # a stable A→B edge present in every year
    return GaussianField(variables=("A", "B", "C", "D"),
                         space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((p, S, T)), resolution="year")


def _noise_field(seed=1, T=12):
    rng = np.random.default_rng(seed)
    p, S = 4, 60
    Z = rng.standard_normal((p, S, T))
    return GaussianField(variables=("A", "B", "C", "D"),
                         space_ids=tuple(f"27{i:05d}"[:7] for i in range(S)),
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((p, S, T)), resolution="year")


def test_persistent_edge_holds_out_of_sample():
    rep = temporal_holdout(_persistent_edge_field(), K=1, holdout_years=2)
    assert rep["ran"] is True
    assert rep["n_train_edges"] >= 1
    assert rep["persistence_rate"] == 1.0        # the real edge recurs with the same sign


def test_short_span_is_skipped_not_asserted():
    rep = temporal_holdout(_persistent_edge_field(T=4), K=1, holdout_years=1, min_train_T=6)
    assert rep["ran"] is False and rep["persistence_rate"] is None


def test_stability_perturbs_time_and_still_recovers_a_strong_edge():
    field = _persistent_edge_field(T=12)
    freq = stability_select(field, K=1, n_subsamples=8, seed=0, perturb_time=True, max_workers=1)
    # the strong A→B edge survives spatial AND temporal subsampling with high frequency
    ab = max((f for (s, t, _), f in freq.items() if {s, t} == {"A", "B"}), default=0.0)
    assert ab >= 0.6
