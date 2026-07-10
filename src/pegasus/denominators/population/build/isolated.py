"""Subprocess-isolated population build — the robust national-scale runner (POP-02).

Runs ``solve_population_tensor_from_sidra_strata`` in a CHILD process with a hard per-attempt timeout,
a process-TREE kill, and a RAM floor, so:

  * the intermittent national GPU segfault (a native torch+polars transition race, exit 139) is
    RECOVERABLE — the parent simply retries past it, and only re-enables national GPU here because the
    isolation makes the crash survivable (``allow_national_gpu=True``);
  * the child's build RAM is released to the OS on every process exit (memory-graceful; the exact
    national peak is measured per build rather than asserted from a state-scale proxy);
  * a pathological hang can never "lose us time" — the parent kills the whole child tree at the deadline.
    A shell ``timeout`` cannot do this reliably on Windows (it does not kill the child's descendants), so
    the parent uses ``taskkill /F /T`` — the actual kill-switch that terminates the process tree (§VII).

Attempt schedule (fast-then-guaranteed): ``max_gpu_attempts`` GPU tries, then ONE CPU try with CUDA
hidden as the guaranteed crash-free fallback. A GPU crash is near-instant, so a retry costs little; a
hang is bounded by the timeout. A *handled* build error (bad inputs, not a crash) is written by the
child as ``status=error`` and NOT retried — only crashes/timeouts/RAM-aborts retry.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IsolatedBuildOutcome:
    status: str                                  # "ok" | "failed"
    output_path: str | None                      # the persisted tensor parquet (values live here, not in the manifest)
    manifest: dict | None                        # PopulationTensorBuild.as_manifest() (metadata only)
    used_gpu: bool                               # whether the winning attempt ran with national GPU enabled
    attempts: tuple[dict, ...] = field(default_factory=tuple)  # per-attempt telemetry
    error: str | None = None                     # child-reported error (handled failure), if any

    def as_manifest(self) -> dict:
        """The build manifest of the winning attempt, annotated with the isolation telemetry — so a
        successful outcome plugs directly into ``resolve_or_build_population_tensor`` as the build result
        (which calls ``build.as_manifest()``). Callers MUST check ``status == 'ok'`` and raise otherwise;
        this never fabricates a manifest for a failed build."""
        return {**(self.manifest or {}), "isolation": {"used_gpu": self.used_gpu, "attempts": list(self.attempts)}}


def _kill_tree(proc: subprocess.Popen) -> None:
    """Terminate the child AND all its descendants. ``proc.kill()`` on Windows kills only the direct
    child; ``taskkill /F /T`` walks the tree. POSIX: kill the process group if we made one, else kill."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=30, check=False,
            )
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _drain(proc: subprocess.Popen) -> None:
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


def run_population_build_isolated(
    *,
    build_kwargs: dict,
    result_stem: str | Path,
    gpu_timeout_s: float = 1200.0,
    cpu_timeout_s: float = 1800.0,
    max_gpu_attempts: int = 2,
    ram_floor_gb: float = 2.0,
    log=None,
) -> IsolatedBuildOutcome:
    """Run the population build in an isolated child, retrying past the intermittent national GPU crash.

    ``build_kwargs`` are the keyword args for ``solve_population_tensor_from_sidra_strata`` (all
    JSON-serializable: paths, ints, bools, mode strings). ``allow_national_gpu`` is set per attempt and
    must NOT be supplied by the caller. ``result_stem`` is a path stem for the per-attempt args/result
    sidecars. ``log`` is an optional ``callable(str)`` for progress lines.
    """
    if "allow_national_gpu" in build_kwargs:
        raise ValueError("allow_national_gpu is controlled by the runner, not the caller")
    _log = log or (lambda _m: None)
    stem = Path(result_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    ram_floor = ram_floor_gb * 1e9
    try:
        import psutil  # optional RAM floor; absent → timeout-only
    except Exception:
        psutil = None

    # attempt plan: GPU (fast, may crash) x N, then CPU (CUDA hidden → guaranteed crash-free) x 1
    plan = [("gpu", True, gpu_timeout_s) for _ in range(max_gpu_attempts)]
    plan.append(("cpu", False, cpu_timeout_s))

    attempts: list[dict] = []
    for i, (label, allow_gpu, timeout_s) in enumerate(plan, start=1):
        arg_path = stem.with_name(f"{stem.name}.attempt{i}.args.json")
        out_path = stem.with_name(f"{stem.name}.attempt{i}.result.json")
        for p in (arg_path, out_path):
            if p.exists():
                p.unlink()
        arg_path.write_text(json.dumps({**build_kwargs, "allow_national_gpu": allow_gpu}), encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["MIMALLOC_PURGE_DELAY"] = "0"
        if not allow_gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""  # hard-hide CUDA on the guaranteed CPU fallback

        _log(f"[attempt {i}/{len(plan)}] mode={label} timeout={timeout_s:.0f}s starting")
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "pegasus.denominators.population.build.isolated",
             "--run", str(arg_path), str(out_path)],
            env=env,
        )
        deadline = time.monotonic() + timeout_s
        outcome_kind = None  # "timeout" | "ram_abort" | None (proc exited on its own)
        while True:
            try:
                rc = proc.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    outcome_kind = "timeout"; _kill_tree(proc); _drain(proc); rc = proc.returncode; break
                if psutil is not None and psutil.virtual_memory().available < ram_floor:
                    outcome_kind = "ram_abort"; _kill_tree(proc); _drain(proc); rc = proc.returncode; break
        dt = round(time.perf_counter() - t0, 1)

        if outcome_kind is None and rc == 0 and out_path.exists():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            if payload.get("status") == "ok":
                attempts.append({"attempt": i, "mode": label, "exit": rc, "seconds": dt, "outcome": "ok"})
                _log(f"[attempt {i}] OK in {dt}s (mode={label})")
                return IsolatedBuildOutcome(
                    status="ok", output_path=payload.get("output_path"), manifest=payload.get("manifest"),
                    used_gpu=allow_gpu, attempts=tuple(attempts),
                )
            # handled build error (not a crash) → do NOT retry, surface it
            err = payload.get("error")
            attempts.append({"attempt": i, "mode": label, "exit": rc, "seconds": dt, "outcome": "error", "error": err})
            _log(f"[attempt {i}] build ERROR (not retrying): {err}")
            return IsolatedBuildOutcome(status="failed", output_path=None, manifest=None,
                                        used_gpu=allow_gpu, attempts=tuple(attempts), error=err)

        # crash (nonzero exit + no result), timeout, or RAM abort → retry the next plan entry
        kind = outcome_kind or f"crash(exit={rc})"
        attempts.append({"attempt": i, "mode": label, "exit": rc, "seconds": dt, "outcome": kind})
        _log(f"[attempt {i}] {kind} after {dt}s → retrying" if i < len(plan) else f"[attempt {i}] {kind} after {dt}s (last attempt)")

    return IsolatedBuildOutcome(status="failed", output_path=None, manifest=None,
                                used_gpu=False, attempts=tuple(attempts),
                                error="all isolated build attempts failed (crash/timeout/ram)")


def _run_child(arg_path: str, out_path: str) -> int:
    """Child entry: deserialize args, run the build, write the result manifest, exit 0. A *handled*
    exception is caught and written as ``status=error`` (still exit 0) so the parent can distinguish it
    from a hard crash — which kills the process with a nonzero exit and leaves no result file."""
    from pegasus.denominators.population.build.orchestrator import solve_population_tensor_from_sidra_strata

    args = json.loads(Path(arg_path).read_text(encoding="utf-8"))
    # Test hook (never present in a real build — the build fn would reject the kwarg): exercise the
    # parent's spawn/timeout/kill-tree/retry machinery with a trivial payload. The attempt index is
    # parsed from the arg filename (`.attemptN.args.json`) so a crash-until-N schedule needs no shared
    # state across the separate attempt processes.
    _tm = args.get("__test_mode__")
    if _tm is not None:
        import re
        m = re.search(r"\.attempt(\d+)\.args\.json$", str(arg_path))
        attempt = int(m.group(1)) if m else 1
        if _tm == "hang":
            time.sleep(10 ** 9)
        if _tm == "crash" or (isinstance(_tm, str) and _tm.startswith("crash_until:") and attempt <= int(_tm.split(":")[1])):
            os._exit(139)  # simulate the native segfault (no result file written)
        Path(out_path).write_text(
            json.dumps({"status": "ok", "output_path": "__test__", "manifest": {"__test__": True, "attempt": attempt}}),
            encoding="utf-8",
        )
        return 0
    try:
        build = solve_population_tensor_from_sidra_strata(**args)
        payload: dict = {"status": "ok", "output_path": build.output_path, "manifest": build.as_manifest()}
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the child
        import traceback
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--run":
        raise SystemExit(_run_child(sys.argv[2], sys.argv[3]))
    raise SystemExit("usage: python -m pegasus.denominators.population.build.isolated --run <args.json> <result.json>")


__all__ = ["IsolatedBuildOutcome", "run_population_build_isolated"]
