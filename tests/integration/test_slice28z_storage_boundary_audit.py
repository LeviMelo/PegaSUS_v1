
from __future__ import annotations

import json
import subprocess
import sys


def test_slice28z_storage_adoption_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28z_storage_boundary_adoption.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
