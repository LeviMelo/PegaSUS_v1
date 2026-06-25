from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("PEGASUS_ACTUAL_STATE_AUDIT_TIMEOUT_SECONDS", "300"))


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _taskkill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _child_main() -> int:
    try:
        from actual_state_panel_runtime import main
        result = main()
        if isinstance(result, dict):
            _emit(result)
        elif result is not None:
            _emit({"classification": "completed_non_dict_result", "result": result})
        return 0
    except SystemExit as exc:
        raise exc
    except Exception as exc:
        _emit({
            "classification": "failed",
            "compile_attempted": False,
            "errors": [
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            ],
            "source": "audit_child_exception",
        })
        return 1


def _supervisor_main() -> int:
    started = time.time()
    env = dict(os.environ)
    env["PEGASUS_ACTUAL_STATE_AUDIT_CHILD"] = "1"

    # Audit-only source timeouts. These do not change production config files.
    env.setdefault("PEGASUS_DATASUS_R_TIMEOUT_SECONDS", "90")
    env.setdefault("PEGASUS_DATASUS_HEARTBEAT_TIMEOUT_SECONDS", "45")

    cmd = [sys.executable, str(Path(__file__).resolve())]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )

    try:
        stdout, stderr = proc.communicate(timeout=DEFAULT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _taskkill_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except Exception:
            stdout, stderr = "", ""

        _emit({
            "classification": "failed_timeout",
            "compile_attempted": None,
            "errors": [
                f"actual_state_panel_audit_timeout_after_{DEFAULT_TIMEOUT_SECONDS}s",
                "The audit child process was killed. This indicates a blocking live-source acquisition or non-terminating compiler stage.",
            ],
            "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "elapsed_seconds": round(time.time() - started, 3),
            "child_pid": proc.pid,
            "child_stdout_tail": stdout[-4000:] if stdout else "",
            "child_stderr_tail": stderr[-4000:] if stderr else "",
            "audit_timeouts": {
                "PEGASUS_DATASUS_R_TIMEOUT_SECONDS": env.get("PEGASUS_DATASUS_R_TIMEOUT_SECONDS"),
                "PEGASUS_DATASUS_HEARTBEAT_TIMEOUT_SECONDS": env.get("PEGASUS_DATASUS_HEARTBEAT_TIMEOUT_SECONDS"),
            },
        })
        return 124

    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()

    if stdout.strip():
        sys.stdout.write(stdout)
        if not stdout.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        _emit({
            "classification": "failed_no_output",
            "compile_attempted": None,
            "errors": [
                f"audit child exited with code {proc.returncode} but produced no stdout JSON",
            ],
            "elapsed_seconds": round(time.time() - started, 3),
            "child_stderr_tail": stderr[-4000:] if stderr else "",
        })

    return int(proc.returncode or 0)


if __name__ == "__main__":
    if os.environ.get("PEGASUS_ACTUAL_STATE_AUDIT_CHILD") == "1":
        raise SystemExit(_child_main())
    raise SystemExit(_supervisor_main())
