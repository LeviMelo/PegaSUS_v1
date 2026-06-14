from __future__ import annotations

from scripts.dev.audits.audit_blockb_she_table_boundary import run_audit


def test_blockb_she_table_boundary_audit_public_api_passes() -> None:
    result = run_audit()
    assert result["audit"] == "blockb_she_table_boundary"
    assert result["status"] == "passed", result
    assert isinstance(result["checked_files"], list)
