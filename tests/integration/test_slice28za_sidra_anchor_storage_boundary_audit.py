from __future__ import annotations

import json
import subprocess
import sys


def test_slice28za_sidra_anchor_storage_audit_passes():
    result = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28za_sidra_anchor_storage_boundary.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
