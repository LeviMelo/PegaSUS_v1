from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()
SLICE = "28ZB"

HSIC_PATH = ROOT / "src" / "pegasus" / "pirs" / "hsic.py"
COMPUTE_RANDOM_PATH = ROOT / "src" / "pegasus" / "compute" / "random.py"
COMPUTE_INIT_PATH = ROOT / "src" / "pegasus" / "compute" / "__init__.py"
AUDIT_PATH = ROOT / "scripts" / "dev" / "audits" / "audit_slice28zb_compute_rng_boundary.py"
UNIT_TEST_PATH = ROOT / "tests" / "unit" / "test_slice28zb_compute_rng_boundary.py"
INTEGRATION_TEST_PATH = ROOT / "tests" / "integration" / "test_slice28zb_compute_rng_boundary_audit.py"
DOC_PATH = ROOT / "docs" / "production_boundaries.md"
UPDATER_TARGET = ROOT / "scripts" / "dev" / "updaters" / "apply_slice28zb_compute_rng_boundary.py"

RFF_FUNCTION = r'''
def _rff_features(torch: Any, values: Any, *, bandwidth: float, features: int, seed: int) -> Any:
    generator = torch_generator(torch, seed=seed, device=values.device)
    omega = torch.randn(features, dtype=values.dtype, device=values.device, generator=generator) / bandwidth
    phase = 2.0 * math.pi * torch.rand(features, dtype=values.dtype, device=values.device, generator=generator)
    return math.sqrt(2.0 / features) * torch.cos(values[:, None] * omega[None, :] + phase[None, :])
'''.lstrip()

NYSTROM_FUNCTION = r'''
def _nystrom_features(torch: Any, values: Any, *, bandwidth: float, landmarks: int, seed: int) -> Any:
    generator = torch_generator(torch, seed=seed, device=values.device)
    indices = torch.randperm(values.shape[0], device=values.device, generator=generator)[:landmarks]
    selected = values[indices]
    cross = torch.exp(-((values[:, None] - selected[None, :]) ** 2) / (2.0 * bandwidth**2))
    basis = torch.exp(-((selected[:, None] - selected[None, :]) ** 2) / (2.0 * bandwidth**2))
    eigenvalues, eigenvectors = torch.linalg.eigh(basis)
    inverse_root = eigenvectors @ torch.diag(torch.clamp(eigenvalues, min=1e-8).rsqrt()) @ eigenvectors.T
    return cross @ inverse_root
'''.lstrip()

TORCH_GENERATOR_FUNCTION = r'''

def torch_generator(torch_module: Any, *, seed: int, device: Any | None = None) -> Any:
    """Create a seeded ``torch.Generator`` through the central compute boundary.

    Domain modules must not construct and seed local generators directly. This
    helper centralizes generator creation so audits can distinguish sanctioned
    compute-boundary seeding from ad hoc numerical state changes.
    """
    seed_int = int(seed)
    try:
        generator = torch_module.Generator(device=device) if device is not None else torch_module.Generator()
    except TypeError:
        generator = torch_module.Generator()
    seed_method = getattr(generator, "manual_seed")
    seed_method(seed_int)
    return generator
'''

AUDIT_SOURCE = r'''
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
'''.lstrip()

UNIT_TEST_SOURCE = r'''
from __future__ import annotations

from pathlib import Path

from pegasus.compute.random import torch_generator


class _FakeGenerator:
    def __init__(self, *, device=None):
        self.device = device
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


class _FakeTorch:
    def Generator(self, device=None):
        return _FakeGenerator(device=device)


def test_slice28zb_torch_generator_uses_central_seed_boundary() -> None:
    generator = torch_generator(_FakeTorch(), seed=20260614, device="cpu")
    assert generator.device == "cpu"
    assert generator.seed == 20260614


def test_slice28zb_hsic_no_longer_seeds_local_generators_directly() -> None:
    source = Path("src/pegasus/pirs/hsic.py").read_text(encoding="utf-8")
    assert "from pegasus.compute.random import torch_generator" in source
    assert "torch_generator(torch, seed=seed, device=values.device)" in source
    assert "generator.manual_seed(" not in source
    assert "torch.Generator(" not in source
'''.lstrip()

INTEGRATION_TEST_SOURCE = r'''
from __future__ import annotations

import json
import subprocess
import sys


def test_slice28zb_compute_rng_boundary_audit_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28zb_compute_rng_boundary.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
'''.lstrip()


def _backup_once(path: Path, suffix: str) -> None:
    if path.exists():
        backup = path.with_name(path.name + suffix)
        if not backup.exists():
            shutil.copy2(path, backup)


def _parse_module(text: str, path: Path) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise RuntimeError(f"Cannot parse {path}: {exc}") from exc


def _replace_top_level_function(text: str, path: Path, name: str, replacement: str) -> str:
    tree = _parse_module(text, path)
    target: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            target = node
            break
    if target is None or target.end_lineno is None:
        raise RuntimeError(f"Could not locate top-level function {name!r} in {path}")
    lines = text.splitlines()
    start = target.lineno - 1
    end = target.end_lineno
    replacement_lines = replacement.rstrip().splitlines()
    new_lines = lines[:start] + replacement_lines + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def _ensure_import(text: str, import_line: str) -> str:
    if import_line in text:
        return text
    lines = text.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_at = idx + 1
    lines.insert(insert_at, import_line)
    return "\n".join(lines).rstrip() + "\n"


def _patch_compute_random() -> None:
    path = COMPUTE_RANDOM_PATH
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    _backup_once(path, ".slice28zb_compute_rng_boundary.bak")
    text = path.read_text(encoding="utf-8")
    text = _ensure_import(text, "from typing import Any")
    if "def torch_generator(" not in text:
        text = text.rstrip() + TORCH_GENERATOR_FUNCTION + "\n"
    path.write_text(text, encoding="utf-8")


def _patch_compute_init() -> None:
    path = COMPUTE_INIT_PATH
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    _backup_once(path, ".slice28zb_compute_rng_boundary.bak")
    text = path.read_text(encoding="utf-8")
    old_import = "from pegasus.compute.random import SeedState, seed_everything"
    new_import = "from pegasus.compute.random import SeedState, seed_everything, torch_generator"
    if old_import in text and new_import not in text:
        text = text.replace(old_import, new_import)
    elif new_import not in text:
        text = _ensure_import(text, new_import)
    if '    "torch_generator",' not in text:
        if '    "seed_everything",' in text:
            text = text.replace('    "seed_everything",', '    "seed_everything",\n    "torch_generator",')
        elif "__all__" in text:
            text = text.rstrip() + '\n# Slice 28ZB export note: torch_generator is intentionally public.\n'
    path.write_text(text, encoding="utf-8")


def _patch_hsic() -> None:
    path = HSIC_PATH
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    _backup_once(path, ".slice28zb_compute_rng_boundary.bak")
    text = path.read_text(encoding="utf-8")
    text = _ensure_import(text, "from pegasus.compute.random import torch_generator")
    text = _replace_top_level_function(text, path, "_rff_features", RFF_FUNCTION)
    text = _replace_top_level_function(text, path, "_nystrom_features", NYSTROM_FUNCTION)
    forbidden = [token for token in ("generator.manual_seed(", "torch.Generator(") if token in text]
    if forbidden:
        raise RuntimeError(f"HSIC compute RNG bypass remains after patch: {forbidden}")
    path.write_text(text, encoding="utf-8")


def _write_auxiliary_files() -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNIT_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTEGRATION_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(AUDIT_SOURCE, encoding="utf-8")
    UNIT_TEST_PATH.write_text(UNIT_TEST_SOURCE, encoding="utf-8")
    INTEGRATION_TEST_PATH.write_text(INTEGRATION_TEST_SOURCE, encoding="utf-8")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    note = dedent(
        """

        ## Slice 28ZB — compute RNG boundary closure

        HSIC random-feature and Nyström generators must be created through
        `pegasus.compute.random.torch_generator`. Domain modules may pass the
        returned generator to PyTorch operations, but may not construct and seed
        `torch.Generator` objects locally. The `audit_slice28zb_compute_rng_boundary.py`
        gate scans production code for direct generator seeding outside the
        central compute boundary.
        """
    ).strip()
    existing = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else "# Production Boundaries\n"
    if "Slice 28ZB — compute RNG boundary closure" not in existing:
        DOC_PATH.write_text(existing.rstrip() + "\n\n" + note + "\n", encoding="utf-8")


def _install_self() -> None:
    UPDATER_TARGET.parent.mkdir(parents=True, exist_ok=True)
    current = Path(__file__).resolve()
    target = UPDATER_TARGET.resolve()
    if current != target:
        UPDATER_TARGET.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")


def _self_audit() -> None:
    # Run the audit logic directly enough to catch patch failures before pytest.
    errors: list[str] = []
    hsic_text = HSIC_PATH.read_text(encoding="utf-8")
    random_text = COMPUTE_RANDOM_PATH.read_text(encoding="utf-8")
    init_text = COMPUTE_INIT_PATH.read_text(encoding="utf-8")
    if "from pegasus.compute.random import torch_generator" not in hsic_text:
        errors.append("hsic.py does not import torch_generator")
    if "torch_generator(torch, seed=seed, device=values.device)" not in hsic_text:
        errors.append("hsic.py does not use torch_generator in feature builders")
    for token in ("generator.manual_seed(", "torch.Generator("):
        if token in hsic_text:
            errors.append(f"hsic.py still contains {token!r}")
    if "def torch_generator(" not in random_text:
        errors.append("compute/random.py does not define torch_generator")
    if "torch_generator" not in init_text:
        errors.append("compute/__init__.py does not export torch_generator")
    if errors:
        raise RuntimeError("Slice 28ZB self-audit failed:\n" + "\n".join(errors))


def main() -> None:
    if not (ROOT / "src" / "pegasus").exists():
        raise RuntimeError("Run this updater from the repository root")
    _patch_compute_random()
    _patch_compute_init()
    _patch_hsic()
    _write_auxiliary_files()
    _install_self()
    _self_audit()
    print(json.dumps({"slice": SLICE, "status": "patched", "files": [str(p.relative_to(ROOT)) for p in (COMPUTE_RANDOM_PATH, COMPUTE_INIT_PATH, HSIC_PATH, AUDIT_PATH, UNIT_TEST_PATH, INTEGRATION_TEST_PATH, DOC_PATH, UPDATER_TARGET)]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
