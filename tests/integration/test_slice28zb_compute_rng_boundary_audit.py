from __future__ import annotations

import json
import subprocess
import sys


def test_slice28zb_compute_rng_boundary_audit_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28zb_compute_rng_boundary.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
