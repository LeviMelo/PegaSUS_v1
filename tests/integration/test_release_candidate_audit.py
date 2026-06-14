from __future__ import annotations

import importlib.util
from pathlib import Path


def _audit_module():
    path = Path("scripts/dev/audits/audit_release_candidate.py")
    spec = importlib.util.spec_from_file_location("audit_release_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_candidate_audit_passes_compact_operational_gates(tmp_path: Path) -> None:
    payload = _audit_module().run_audit(tmp_path)
    assert payload["status"] == "passed", payload
    assert payload["classification"] == "fixture_validated"
    assert payload["error_count"] == 0
    assert all(payload["checks"].values())
