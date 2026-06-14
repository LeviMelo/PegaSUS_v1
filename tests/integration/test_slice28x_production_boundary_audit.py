from __future__ import annotations

from scripts.dev.audits.audit_slice28x_production_boundaries import run_audit


def test_slice28x_production_boundary_audit_runs():
    result = run_audit()
    assert result["status"] == "passed"
    assert result["audit"] == "slice28x_production_boundaries"
