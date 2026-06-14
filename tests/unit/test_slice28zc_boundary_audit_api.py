from __future__ import annotations

from scripts.dev.audits.audit_slice28x_production_boundaries import run_audit


def test_slice28zc_slice28x_audit_preserves_run_audit_import_api() -> None:
    result = run_audit()
    assert result["audit"] == "slice28x_production_boundaries"
    assert result["status"] == "passed"
    assert result["storage_bypass_count"] == 0
    assert result["compute_bypass_count"] == 0
    assert result["error_count"] == 0
