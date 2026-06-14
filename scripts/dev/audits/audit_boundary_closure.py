from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

AUDIT_SCRIPTS = [
    "scripts/dev/audits/audit_slice28x_production_boundaries.py",
    "scripts/dev/audits/audit_slice28z_storage_boundary_adoption.py",
    "scripts/dev/audits/audit_slice28za_sidra_anchor_storage_boundary.py",
    "scripts/dev/audits/audit_slice28zb_compute_rng_boundary.py",
]

MAX_STDOUT_TAIL = 800
MAX_STDERR_TAIL = 800


def _load_last_json(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        return {"status": "failed", "error_count": 1, "errors": ["audit produced no stdout"]}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        # Keep diagnostics bounded. Do not embed unbounded audit output in the
        # aggregate audit payload.
        return {
            "status": "failed",
            "error_count": 1,
            "errors": ["audit stdout was not a single JSON document"],
            "stdout_tail": stripped[-MAX_STDOUT_TAIL:],
        }


def _result_summary(script: str, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload = _load_last_json(proc.stdout)
    status = payload.get("status", "failed")
    if proc.returncode != 0 and status == "passed":
        status = "failed"
    summary: dict[str, Any] = {
        "script": script,
        "status": status,
        "returncode": proc.returncode,
        "storage_bypass_count": payload.get("storage_bypass_count"),
        "compute_bypass_count": payload.get("compute_bypass_count"),
        "error_count": payload.get("error_count", len(payload.get("errors", [])) if isinstance(payload.get("errors"), list) else None),
        "errors_truncated": payload.get("errors_truncated", 0),
    }
    errors = payload.get("errors", [])
    if errors:
        summary["errors"] = errors[:5] if isinstance(errors, list) else [str(errors)]
    if proc.stderr.strip():
        summary["stderr_tail"] = proc.stderr.strip()[-MAX_STDERR_TAIL:]
    if "stdout_tail" in payload:
        summary["stdout_tail"] = payload["stdout_tail"]
    return summary


def _run_audit(script: str) -> dict[str, Any]:
    path = ROOT / script
    if not path.exists():
        return {
            "script": script,
            "status": "failed",
            "returncode": None,
            "error_count": 1,
            "errors": [f"audit script missing: {script}"],
        }
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return _result_summary(script, proc)


def main() -> None:
    results = [_run_audit(script) for script in AUDIT_SCRIPTS]
    failures = [result for result in results if result.get("status") != "passed" or result.get("returncode") not in (0, None)]
    payload = {
        "audit": "boundary_closure",
        "status": "passed" if not failures else "failed",
        "scripts": AUDIT_SCRIPTS,
        "result_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "errors": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
