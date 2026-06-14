from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HSIC_PATH = ROOT / "src" / "pegasus" / "pirs" / "hsic.py"
COMPUTE_RANDOM_PATH = ROOT / "src" / "pegasus" / "compute" / "random.py"

FORBIDDEN_GLOBAL_TOKENS = (
    "generator.manual_seed(",
    "torch.Generator(",
)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "\\")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not HSIC_PATH.exists():
        errors.append(f"missing target: {_rel(HSIC_PATH)}")
    else:
        hsic_text = HSIC_PATH.read_text(encoding="utf-8")
        if "from pegasus.compute.random import torch_generator" not in hsic_text:
            errors.append("pirs/hsic.py must import torch_generator from the compute boundary")
        if "torch_generator(torch, seed=seed, device=values.device)" not in hsic_text:
            errors.append("pirs/hsic.py must route HSIC random-feature generators through torch_generator")

    if not COMPUTE_RANDOM_PATH.exists():
        errors.append(f"missing target: {_rel(COMPUTE_RANDOM_PATH)}")
    else:
        random_text = COMPUTE_RANDOM_PATH.read_text(encoding="utf-8")
        if "def torch_generator(" not in random_text:
            errors.append("compute/random.py must define torch_generator")

    scanned: list[str] = []
    for path in sorted((ROOT / "src" / "pegasus").rglob("*.py")):
        rel = _rel(path)
        if rel == "src\\pegasus\\compute\\random.py":
            continue
        text = path.read_text(encoding="utf-8")
        scanned.append(rel)
        for token in FORBIDDEN_GLOBAL_TOKENS:
            if token in text:
                errors.append(f"compute RNG bypass candidate: {rel}: contains {token!r}")

    payload = {
        "audit": "slice28zb_compute_rng_boundary",
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "scanned_files": len(scanned),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
