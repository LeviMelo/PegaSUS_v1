"""Deterministic seed harmonization across Python, NumPy, and PyTorch."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedState:
    seed: int
    numpy_seeded: bool
    torch_seeded: bool
    deterministic_algorithms: bool

    def as_manifest(self) -> dict[str, Any]:
        return dict(vars(self))


def seed_everything(seed: int, *, torch_module: Any = None, deterministic: bool = True) -> SeedState:
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
        numpy_seeded = True
    except ImportError:
        pass
    torch_seeded = False
    torch = torch_module
    if torch is None:
        try:
            import torch as imported_torch

            torch = imported_torch
        except ImportError:
            torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch_seeded = True
    return SeedState(seed, numpy_seeded, torch_seeded, deterministic and torch_seeded)

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

