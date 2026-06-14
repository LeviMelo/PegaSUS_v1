from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_slice28zc_boundary_closure_audit_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_boundary_closure.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert payload["failure_count"] == 0
    assert not payload["errors"]


def test_slice28zc_slice28x_audit_is_strict_for_guarded_files() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28x_production_boundaries.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads(proc.stdout)
    assert payload["status"] == "passed"
    assert payload["storage_bypass_count"] == 0
    assert payload["compute_bypass_count"] == 0
    assert payload["error_count"] == 0
    assert payload["errors"] == []
