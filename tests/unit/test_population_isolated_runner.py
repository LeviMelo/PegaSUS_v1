"""POP-02 subprocess-isolated population build — the robustness contract.

The national GPU build intermittently hard-segfaults from a native torch+polars transition race; the
isolated runner must RECOVER (retry past the crash), HARD-TERMINATE a pathological hang (kill the child
tree, never orphan it), and fall back to a guaranteed crash-free CPU attempt. These are exercised through
the child's ``__test_mode__`` hook (attempt index parsed from the arg filename) so the contract is proven
without a heavy real build. One focused proof-of-capability test, not a battery.
"""

from __future__ import annotations

import time

from pegasus.denominators.population.build.isolated import (
    IsolatedBuildOutcome,
    run_population_build_isolated,
)


def test_crash_then_cpu_fallback_recovers(tmp_path):
    """GPU attempts that segfault (exit 139) are retried; the CPU fallback then succeeds — the national
    segfault-recovery contract. Outcomes record the two crashes and the win."""
    out = run_population_build_isolated(
        build_kwargs={"__test_mode__": "crash_until:2"},
        result_stem=tmp_path / "run",
        gpu_timeout_s=30, cpu_timeout_s=30, max_gpu_attempts=2,
    )
    assert out.status == "ok"
    assert out.used_gpu is False  # the CPU fallback (3rd attempt) is what succeeded
    assert [a["outcome"] for a in out.attempts] == ["crash(exit=139)", "crash(exit=139)", "ok"]


def test_first_gpu_attempt_succeeds(tmp_path):
    """When the GPU attempt does not crash, it wins immediately — no wasted CPU fallback."""
    out = run_population_build_isolated(
        build_kwargs={"__test_mode__": "crash_until:0"},
        result_stem=tmp_path / "run",
        gpu_timeout_s=30, cpu_timeout_s=30, max_gpu_attempts=2,
    )
    assert out.status == "ok" and out.used_gpu is True and len(out.attempts) == 1


def test_hang_is_hard_terminated(tmp_path):
    """A pathological hang is killed at the deadline (tree-kill), every attempt times out, and the run
    fails cleanly rather than blocking forever — the hard-terminator contract."""
    t0 = time.perf_counter()
    out = run_population_build_isolated(
        build_kwargs={"__test_mode__": "hang"},
        result_stem=tmp_path / "run",
        gpu_timeout_s=3, cpu_timeout_s=3, max_gpu_attempts=1,
    )
    elapsed = time.perf_counter() - t0
    assert out.status == "failed"
    assert [a["outcome"] for a in out.attempts] == ["timeout", "timeout"]
    # The run is BOUNDED (never a runaway): two 3s deadlines + a per-attempt kill/drain capped at 30s.
    # The bound is generous so the test is not flaky under heavy concurrent load (child spawn + process
    # reaping stretch when the box is busy) — the point is termination, not a tight wall-time.
    assert elapsed < 90, f"tree-kill did not bound the run ({elapsed:.1f}s)"


def test_caller_may_not_set_allow_national_gpu(tmp_path):
    """The GPU policy is the runner's to set per attempt, not the caller's."""
    import pytest

    with pytest.raises(ValueError):
        run_population_build_isolated(
            build_kwargs={"allow_national_gpu": True}, result_stem=tmp_path / "run",
        )


def test_outcome_as_manifest_carries_isolation_telemetry():
    """A successful outcome plugs into resolve_or_build_population_tensor via as_manifest(), annotated
    with the isolation telemetry (which attempt/mode won)."""
    o = IsolatedBuildOutcome(
        status="ok", output_path="x", manifest={"tensor_shape": [1, 2, 3, 4, 5]},
        used_gpu=True, attempts=({"attempt": 1, "outcome": "ok"},),
    )
    man = o.as_manifest()
    assert man["tensor_shape"] == [1, 2, 3, 4, 5]
    assert man["isolation"]["used_gpu"] is True
    assert man["isolation"]["attempts"][0]["outcome"] == "ok"
