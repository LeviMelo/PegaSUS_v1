"""Maternal-child compatibility attach boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7
from pegasus.she.maternal_child import summarize_maternal_child_events


def attach_maternal_child_compile_fields(
    *,
    run_dir: str | Path,
    sinasc_events_path: str | Path,
    municipality_cod6: str | None = None,
    datasus_uf_prefix: str = "27",
) -> dict[str, Any]:
    if municipality_cod6 is not None:
        municipality_cod7 = datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)
    else:
        municipality_cod7 = None
        # State-level maternal-child attachment requires multi-municipality
        # support and must remain descriptive until a population denominator is attached.
    summary = summarize_maternal_child_events(sinasc_events_path, datasus_uf_prefix=datasus_uf_prefix)
    return {
        "run_dir": str(run_dir),
        "municipality_cod6": municipality_cod6,
        "municipality_cod7": municipality_cod7,
        "summary": summary.__dict__,
    }


__all__ = ["attach_maternal_child_compile_fields"]
