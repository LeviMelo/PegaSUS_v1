from __future__ import annotations

from scripts.dev.audits.audit_slice28y_efg_semantic_manifest import run_audit


def test_slice28y_semantic_manifest_audit_passes():
    result = run_audit()
    assert result["status"] == "passed"
    assert result["audit"] == "slice28y_efg_semantic_manifest"
