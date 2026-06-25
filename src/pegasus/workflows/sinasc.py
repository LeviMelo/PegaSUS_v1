from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.datasus.sinasc_normalize import normalize_sinasc_events
from pegasus.output.sinasc_efg_bundle import write_sinasc_fixture_efg_bundle
from pegasus.output.validate import validate_output_bundle


def run_datasus_normalize_sinasc(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_manifest_hash: str,
) -> dict[str, Any]:
    return normalize_sinasc_events(
        input_path=input_path,
        output_path=output_path,
        source_manifest_hash=source_manifest_hash,
    )


def run_build_sinasc_fixture(
    *,
    sinasc_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
    datasus_uf_prefix: str | None = None,
) -> dict[str, Any]:
    output = write_sinasc_fixture_efg_bundle(
        sinasc_events_path=sinasc_events_path,
        run_dir=run_dir,
        municipality_cod6=municipality_cod6,
        datasus_uf_prefix=datasus_uf_prefix,
    )
    validation = validate_output_bundle(run_dir=str(output))
    return {"run_dir": output, "validation": validation}
