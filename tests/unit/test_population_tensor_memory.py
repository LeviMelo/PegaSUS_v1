"""POP-02 M1: PopulationTensorProblem stores numpy arrays (not Python tuples), None→NaN.

The §V.1 memory contract — the O(n_cells) inputs must not materialize as ~32 B/element Python
float tuples (they OOM a 32 GB box at national scale). This pins the representation + the NaN
sentinel semantics so the solver/loss can read the arrays directly.
"""

from __future__ import annotations

import numpy as np

from pegasus.she.reconstruction.schema import PopulationTensorProblem


def test_array_fields_are_numpy_with_nan_sentinels() -> None:
    prob = PopulationTensorProblem(
        shape=(2, 1, 1, 1, 1),
        anchors=[10.0, None],                 # None element → NaN
        closure_totals=[10.0, None],
        births=None,                           # whole-field None stays None
    )
    assert isinstance(prob.anchors, np.ndarray) and prob.anchors.dtype == np.float64
    assert prob.anchors[0] == 10.0 and np.isnan(prob.anchors[1])
    assert isinstance(prob.closure_totals, np.ndarray) and np.isnan(prob.closure_totals[1])
    assert prob.births is None                 # not fabricated into an array


def test_no_python_tuple_storage_at_scale() -> None:
    # A 200k-cell problem's anchor array is a compact float64 buffer (~1.6 MB), NOT a Python
    # tuple of 200k boxed floats (~6.4 MB + per-object overhead). This is the representation that
    # makes the 10^8-cell national tensor feasible.
    n = 200_000
    prob = PopulationTensorProblem(shape=(n, 1, 1, 1, 1), anchors=list(range(n)))
    assert isinstance(prob.anchors, np.ndarray)
    assert prob.anchors.nbytes == n * 8        # dense float64, not object-boxed
    assert not isinstance(prob.anchors, tuple)


def test_accepts_tuple_list_or_array_interchangeably() -> None:
    # Backward compatibility: existing callers pass tuples; new callers pass arrays. Both normalize.
    from_tuple = PopulationTensorProblem(shape=(3, 1, 1, 1, 1), anchors=(1.0, 2.0, 3.0))
    from_array = PopulationTensorProblem(shape=(3, 1, 1, 1, 1), anchors=np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(from_tuple.anchors, from_array.anchors)
    assert isinstance(from_tuple.anchors, np.ndarray)
