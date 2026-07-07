"""RES-01 / §II.7 — the panel→coarse→fine multiresolution LDO wired as a run_ldo drop-in.

`run_ldo_multiresolution` must (1) spatially coarsen the fine field by cod6 prefix into a cheap
discovery grain, and (2) run the two-pass coarse→fine LDO, returning a merged LDORun the caller
consumes exactly like `run_ldo`. The coarsening itself must reduce the spatial-unit count while
preserving variable/time axes.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.margins import GaussianField
from pegasus.ldo.orchestrator import LDORun, run_ldo_multiresolution
from pegasus.ldo.resolution import coarsen_field_spatial


def _field_over_ufs(seed: int = 0):
    rng = np.random.default_rng(seed)
    # 3 UFs × 8 munis each, cod6 = UF(2 digits) + 4-digit muni.
    space_ids = tuple(f"{uf}{m:04d}" for uf in ("11", "22", "33") for m in range(8))
    S, T = len(space_ids), 30
    Z = rng.standard_normal((4, S, T))
    Z[1] += 0.7 * Z[0]          # a real A→B edge
    return GaussianField(variables=("A", "B", "C", "D"), space_ids=space_ids,
                         time_ids=tuple(range(T)), Z=Z, W=np.ones((4, S, T)), resolution="year")


def test_coarsen_field_spatial_pools_by_uf_prefix():
    field = _field_over_ufs()
    coarse = coarsen_field_spatial(field, level=2)
    assert coarse.space_ids == ("11", "22", "33")          # 24 munis → 3 UFs
    assert coarse.Z.shape == (4, 3, 30)                     # variable/time axes preserved
    assert coarse.variables == field.variables
    assert np.isfinite(coarse.Z).all()


def test_coarsen_preserves_weighted_group_mean():
    field = _field_over_ufs()
    coarse = coarsen_field_spatial(field, level=2)
    # UF "11" is the first 8 munis; its coarse value == mean over those munis (unit weights).
    expected = field.Z[:, :8, :].mean(axis=1)
    assert np.allclose(coarse.Z[:, 0, :], expected, atol=1e-9)


def test_run_ldo_multiresolution_returns_merged_ldorun():
    field = _field_over_ufs()
    run = run_ldo_multiresolution(field, K=2, coarse_level=2, seed=0)
    assert isinstance(run, LDORun)
    assert "multiresolution" in run.diagnostics
    mr = run.diagnostics["multiresolution"]
    assert "candidates" in mr and "sensitivity_screen" in mr
    # the planted A→B edge is among the discovered links (directed or contemporaneous)
    pairs = {(r.source_var, r.target_var) for r in run.link_records}
    assert ("A", "B") in pairs or ("B", "A") in pairs
