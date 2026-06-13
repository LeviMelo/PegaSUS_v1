
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.run_candidates import attach_pirs_candidate_gate_to_run, write_pirs_candidate_manifest


def run_build_pirs_candidates_from_run(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return write_pirs_candidate_manifest(run_dir=run_dir, output=output)


def run_attach_pirs_candidate_gate(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return attach_pirs_candidate_gate_to_run(run_dir=run_dir, output=output)
