from __future__ import annotations

from pathlib import Path

from pegasus.datasus.cnes_normalize import normalize_cnes_st_events
from pegasus.datasus.sih_normalize import normalize_sih_rd_events
from pegasus.output.cnes_sih_efg_bundle import write_cnes_sih_fixture_efg_bundle
from pegasus.output.validate import validate_output_bundle


def run_datasus_normalize_cnes(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, object]:
    return normalize_cnes_st_events(input_path=input_path, output_path=output_path, source_manifest_hash=source_manifest_hash)


def run_datasus_normalize_sih(*, input_path: str | Path, output_path: str | Path, source_manifest_hash: str) -> dict[str, object]:
    return normalize_sih_rd_events(input_path=input_path, output_path=output_path, source_manifest_hash=source_manifest_hash)


def run_build_cnes_sih_fixture(*, cnes_events_path: str | Path, sih_events_path: str | Path, run_dir: str | Path, municipality_cod6: str | None = None) -> dict[str, object]:
    run = write_cnes_sih_fixture_efg_bundle(cnes_events_path=cnes_events_path, sih_events_path=sih_events_path, run_dir=run_dir, municipality_cod6=municipality_cod6)
    validation = validate_output_bundle(run_dir=str(run))
    return {"run_dir": run, "validation": validation}
