from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    production_modules = (
        "src/pegasus/compute/devices.py",
        "src/pegasus/storage/parquet.py",
        "src/pegasus/she/population/sparse_admm.py",
        "src/pegasus/she/population/state_space.py",
        "src/pegasus/geo/geodata.py",
        "src/pegasus/she/stdfm/pipeline.py",
        "src/pegasus/pirs/hsic.py",
        "src/pegasus/datasus/client_microdatasus.py",
        "src/pegasus/workflows/ingest_datasus.py",
        "src/pegasus/workflows/ingest_sidra.py",
        "src/pegasus/acceptance/contracts.py",
    )
    for relative in production_modules:
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        text = _text(relative)
        if "slice0_scaffold_only" in text or "Slice 0 scaffold module" in text:
            errors.append(f"scaffold_marker:{relative}")
        try:
            ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"syntax:{relative}:{exc}")

    registry = yaml.safe_load(_text("config/registries/population_solver_registry.yaml"))
    entries = {str(item.get("id")): item for item in registry.get("entries", [])}
    for solver_id in ("sparse_block_coordinate_v1", "sparse_block_coordinate_sim_informed_v1"):
        if not str(entries.get(solver_id, {}).get("status", "")).startswith("active"):
            errors.append(f"population_solver_not_active:{solver_id}")

    docs = (
        "architecture.md", "efg.md", "she.md", "pirs.md", "output_bundle.md",
        "validation.md", "compute_backend.md", "datasus_subsystem.md",
        "sidra_subsystem.md", "development_slices.md", "level3_acceptance.md",
    )
    for name in docs:
        text = _text(f"docs/{name}")
        if "scaffold placeholder" in text.lower() or len(text.strip()) < 100:
            errors.append(f"documentation_incomplete:{name}")

    acceptance_text = _text("src/pegasus/acceptance/contracts.py")
    for number in range(1, 24):
        if f'"{number:02d}_' not in acceptance_text:
            errors.append(f"level3_check_missing:{number:02d}")

    payload = {"audit": "slices28a36a_project_completion", "ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
