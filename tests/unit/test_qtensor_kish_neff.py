"""The live compile_attach n_eff uses the §3.12.3 Kish effective size (refactor §4.2).

Before this fix, ``efg/compile_attach.py::_vector_diagnostics`` computed
``n_eff = n_obs * (1-I)/(1+I)`` over a plain non-null cell count — non-conformant with
MSD-I §3.12.3, which ``efg/q_tensor.py::compute_q_state`` already implemented correctly
(Kish ``(Σw)²/Σw²`` deflated by ``1/(1+max(0,I))``). The live compile path now shares that
single source of truth. These tests pin the live path to the Kish base so it cannot silently
drift back to the raw-count estimator.
"""

from __future__ import annotations

from pegasus.efg.compile_attach import _vector_diagnostics
from pegasus.efg.q_tensor import _kish_effective_n


def test_vector_diagnostics_uses_kish_not_raw_observed_count() -> None:
    # Concentrated mass: 3 non-null cells but nearly all weight in one -> Kish << n_obs.
    # (102)^2 / (100^2 + 1 + 1) ~= 1.04, whereas the old n_obs estimator returned 3.0.
    vector = [100.0, 1.0, 1.0]
    diag = _vector_diagnostics(vector, panel=None, adjacency=None)
    kish = _kish_effective_n([100.0, 1.0, 1.0])
    assert kish is not None and kish < 1.1
    # No panel -> moran_i is None -> no deflation, so n_eff is exactly the Kish base.
    assert abs(diag["n_eff"] - kish) < 1e-9
    assert diag["n_eff"] < 2.0  # decisively NOT the old raw-count value of 3.0


def test_uniform_weights_recover_the_cell_count() -> None:
    # Under equal weights Kish equals the cell count: the estimator is a strict
    # generalization of the old raw-count behaviour, not a regression for flat fields.
    diag = _vector_diagnostics([5.0, 5.0, 5.0, 5.0], panel=None, adjacency=None)
    assert abs(diag["n_eff"] - 4.0) < 1e-9
