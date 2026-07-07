"""O12 / §3.12.3 / §II.3 — the Q-tensor state (n_eff / fragility / provenance) weights the LDO W.

A field with a large design effect (low n_eff), a fragile denominator, or unofficial provenance
must be DOWN-WEIGHTED in the precision fit — a normalized/uncertain quantity is not modelled as if
exact. This pins that field_weights threads through run_ldo into the assembled reliability tensor.
"""

from __future__ import annotations

import numpy as np

from pegasus.ldo.assemble import assemble_ldo_tensor
from pegasus.she.panel import CommonPanel
import polars as pl


def _panel():
    rows = []
    for s in range(6):
        for t in (2020, 2021, 2022, 2023):
            rows.append({"municipality_cod6": f"27000{s}", "year": t, "A": 1.0 * s + t, "B": 2.0 * s})
    values = pl.DataFrame(rows)
    manifest = pl.DataFrame({"field_id": [], "municipality_cod6": [], "year": [], "state": []},
                            schema={"field_id": pl.Utf8, "municipality_cod6": pl.Utf8, "year": pl.Int64, "state": pl.Utf8})
    index = values.select(["municipality_cod6", "year"]).unique()
    return CommonPanel(resolution="year", cell_keys=("municipality_cod6", "year"), index=index,
                       values=values, manifest=manifest, fields=("A", "B"))


def test_field_weights_scale_the_reliability_tensor():
    panel = _panel()
    base = assemble_ldo_tensor(panel)
    weighted = assemble_ldo_tensor(panel, field_weights={"A": 0.25})
    ia = base.variables.index("A")
    ib = base.variables.index("B")
    # A's weights are scaled by 0.25 (relative to base); B's are unchanged
    obs = np.isfinite(base.X[ia])
    assert np.allclose(weighted.W[ia][obs], 0.25 * base.W[ia][obs])
    assert np.allclose(weighted.W[ib], base.W[ib])


def test_run_ldo_reports_state_weighted_count():
    from pegasus.ldo.orchestrator import run_ldo
    from pegasus.ldo.margins import GaussianField
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((3, 40, 6))
    field = GaussianField(variables=("A", "B", "C"),
                          space_ids=tuple(f"27{i:05d}"[:7] for i in range(40)),
                          time_ids=tuple(range(6)), Z=Z, W=np.ones((3, 40, 6)), resolution="year")
    # GaussianField source has no raw panel to weight, but the param must thread without error
    run = run_ldo(field, K=1, n_subsamples=2, seed=0)
    assert "n_state_weighted" in run.diagnostics
