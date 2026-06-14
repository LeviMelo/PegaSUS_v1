from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

STORAGE_PATTERNS = re.compile(r"pyarrow\.parquet|\bpq\.read_table\b|\bpq\.write_table\b|\bpl\.read_parquet\b|\bpl\.scan_parquet\b|\.write_parquet\(")
COMPUTE_PATTERNS = re.compile(r"torch\.cuda|torch\.device|manual_seed|np\.random\.seed|random\.seed")

STRICT_STORAGE_FILES = [
    "src/pegasus/efg/compile_attach.py",
    "src/pegasus/efg/promotion_apply.py",
    "src/pegasus/output/cnes_sih_compile_attach.py",
    "src/pegasus/output/maternal_child_compile_attach.py",
    "src/pegasus/output/population_tensor_compile_attach.py",
    "src/pegasus/output/race_bridge_attach.py",
    "src/pegasus/output/sidra_denominator_anchor.py",
    "src/pegasus/dashboard/read_only.py",
    "src/pegasus/acceptance/contracts.py",
]

STRICT_COMPUTE_FILES = [
    "src/pegasus/pirs/hsic.py",
    "src/pegasus/pirs/nulls.py",
    "src/pegasus/pirs/nystrom.py",
    "src/pegasus/pirs/rff.py",
    "src/pegasus/she/stdfm/torch_solver.py",
    "src/pegasus/she/population/torch_kernels.py",
    "src/pegasus/she/population/block_coordinate.py",
    "src/pegasus/she/population/sparse_admm.py",
]

REQUIRED_UNBLOCKED = [
    "src/pegasus/efg/core_seed.py",
    "src/pegasus/efg/bridges.py",
    "src/pegasus/registries/generic.py",
    "src/pegasus/registries/models.py",
    "src/pegasus/registries/residuals.py",
    "src/pegasus/registries/hsic.py",
    "src/pegasus/registries/nulls.py",
    "src/pegasus/registries/output.py",
    "src/pegasus/registries/events.py",
    "src/pegasus/registries/composite_decoders.py",
    "src/pegasus/registries/race_axis.py",
    "src/pegasus/registries/sidra.py",
    "src/pegasus/registries/manifest.py",
]


def _scan(paths: list[str], pattern: re.Pattern[str]) -> list[dict]:
    findings: list[dict] = []
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.search(line):
                findings.append({"path": rel, "line": number, "text": line.strip()})
    return findings


def run_audit() -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_UNBLOCKED:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"required file missing: {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if "slice0_scaffold_only" in text or "BlockedModuleError" in text:
            errors.append(f"required module still scaffold-blocked: {rel}")

    storage_hits = _scan(STRICT_STORAGE_FILES, STORAGE_PATTERNS)
    compute_hits = _scan(STRICT_COMPUTE_FILES, COMPUTE_PATTERNS)

    # This first consolidation slice records production-boundary bypasses as
    # warnings rather than blocking every historical module at once.  The audit
    # is still useful because new/de-scaffolded files above are hard errors, and
    # future slices can promote these warnings to errors as files are migrated.
    for hit in storage_hits:
        warnings.append(f"storage bypass candidate: {hit['path']}:{hit['line']}: {hit['text']}")
    for hit in compute_hits:
        warnings.append(f"compute bypass candidate: {hit['path']}:{hit['line']}: {hit['text']}")

    return {
        "audit": "slice28x_production_boundaries",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "storage_bypass_count": len(storage_hits),
        "compute_bypass_count": len(compute_hits),
    }


def main() -> None:
    result = run_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
