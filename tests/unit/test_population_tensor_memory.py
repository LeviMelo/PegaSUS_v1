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


def test_blocked_solve_equals_whole_problem_exactly() -> None:
    """POP-02 M3: locality-blocked solve is numerically EXACT vs whole-problem (the objective is
    locality-separable when migration_totals is None), while bounding peak memory to O(block)."""
    from pegasus.she.reconstruction.solvers import (
        solve_population_tensor_blocked,
        solve_population_tensor_problem,
    )

    # 8 localities x 1 period x 3 ages: anchors + per-locality closure totals + age smoothness.
    s, t, a = 8, 1, 3
    rng = np.random.RandomState(0)  # noqa: NPY002 - fixed seed, test only
    anchors = rng.uniform(50, 200, size=s * t * a)
    closure = np.array([float(anchors[i * a:(i + 1) * a].sum()) for i in range(s)])
    prob = PopulationTensorProblem(
        shape=(s, t, a, 1, 1),
        anchors=anchors,
        closure_totals=closure,
        initial_population=anchors.copy(),
        mode="independent_denominator",
    )

    whole = solve_population_tensor_problem(prob, max_iterations=2000, tolerance=1e-8)
    # block_target_cells tiny → forces ~2 localities/block (multiple blocks), exercising the split.
    blocked = solve_population_tensor_blocked(prob, max_iterations=2000, tolerance=1e-8, block_target_cells=6)

    assert len(blocked.population) == s * t * a
    np.testing.assert_allclose(
        np.asarray(blocked.population), np.asarray(whole.population), rtol=1e-6, atol=1e-4
    )
    # telemetry records the block structure
    assert blocked.telemetry.objective_terms.get("n_blocks", 0) >= 2
