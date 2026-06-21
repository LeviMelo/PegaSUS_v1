from __future__ import annotations

import inspect

from pegasus.output import maternal_child_compile_attach as mc


def test_maternal_child_attach_accepts_optional_municipality_cod6() -> None:
    signature = inspect.signature(mc.attach_maternal_child_compile_fields)
    annotation = signature.parameters["municipality_cod6"].annotation
    assert annotation in {"str | None", str | None}


def test_maternal_child_attach_has_state_level_branch() -> None:
    source = inspect.getsource(mc.attach_maternal_child_compile_fields)

    assert "if municipality_cod6 is not None:" in source
    assert "else:" in source
    assert "State-level maternal-child attachment requires multi-municipality" in source
    assert "datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)" in source
